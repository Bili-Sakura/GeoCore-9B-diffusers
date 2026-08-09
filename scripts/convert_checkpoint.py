"""Convert a GeoCore-9B training checkpoint into a self-contained Diffusers model directory.

    python scripts/convert_checkpoint.py \
        --ckpt exps/GeoCore-9B_new_rev/checkpoints/0300000.pt \
        --out huggingface/

The export bundles remote-code copies of the transformer, scheduler, and pipeline from
`src/diffusers` so Hub / local inference needs only Diffusers — not `models/flux2.py` or the
training codebase.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
MODEL_SIZES = ("9b", "4b")
ZERO_INIT_PROBES = (
    "final_layer.linear.weight",
    "time_in.out_layer.weight",
    "double_stream_modulation_img.lin.weight",
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIFFUSERS = os.path.join(REPO_ROOT, "src", "diffusers")
TEMPLATE_DIR = os.path.join(REPO_ROOT, "huggingface")

MODEL_INDEX = {
    "_class_name": ["pipeline", "GeoCorePipeline"],
    "_diffusers_version": "0.32.0",
    "scheduler": ["scheduling_geocore", "GeoCoreFlowMatchEulerScheduler"],
    "transformer": ["transformer_geocore", "GeoCoreTransformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLFlux2"],
}


def strip_wrapper_prefixes(name: str) -> str:
    while name.startswith(("module.", "_orig_mod.")):
        name = name.split(".", 1)[1]
    return name


def shard_state_dict(state_dict, max_shard_bytes: int):
    shards, current, current_bytes = [], {}, 0
    for key, tensor in state_dict.items():
        size = tensor.numel() * tensor.element_size()
        if current and current_bytes + size > max_shard_bytes:
            shards.append(current)
            current, current_bytes = {}, 0
        current[key] = tensor
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def write_transformer_shards(state_dict: dict, out_dir: str, max_shard_bytes: int) -> None:
    transformer_dir = os.path.join(out_dir, "transformer")
    os.makedirs(transformer_dir, exist_ok=True)
    total_bytes = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
    shards = shard_state_dict(state_dict, max_shard_bytes)
    shard_count = len(shards)
    weight_map = {}
    for index, shard in enumerate(shards, start=1):
        name = "diffusion_pytorch_model.safetensors" if shard_count == 1 else (
            f"diffusion_pytorch_model-{index:05d}-of-{shard_count:05d}.safetensors"
        )
        save_file(shard, os.path.join(transformer_dir, name), metadata={"format": "pt"})
        for key in shard:
            weight_map[key] = name
        print(f"  wrote transformer/{name} ({len(shard)} tensors)")
    if shard_count > 1:
        with open(os.path.join(transformer_dir, "diffusion_pytorch_model.safetensors.index.json"), "w") as handle:
            json.dump({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, handle, indent=2)


def make_self_contained_repo(out_dir: str, model_size: str, dtype_name: str, weights: str, steps: int | None) -> None:
    """Copy Diffusers sources into the model repo for trust_remote_code loading."""
    os.makedirs(os.path.join(out_dir, "transformer"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "scheduler"), exist_ok=True)

    shutil.copy2(
        os.path.join(SRC_DIFFUSERS, "pipelines", "geocore", "pipeline_geocore.py"),
        os.path.join(out_dir, "pipeline.py"),
    )
    shutil.copy2(
        os.path.join(SRC_DIFFUSERS, "models", "transformers", "transformer_geocore.py"),
        os.path.join(out_dir, "transformer", "transformer_geocore.py"),
    )
    shutil.copy2(
        os.path.join(SRC_DIFFUSERS, "schedulers", "scheduling_geocore.py"),
        os.path.join(out_dir, "scheduler", "scheduling_geocore.py"),
    )

    # Drop legacy remote-code / bootstrap bundles if regenerating an older export.
    for legacy in (
        os.path.join(out_dir, "bootstrap_geocore.py"),
        os.path.join(out_dir, "transformer", "modeling_geocore_transformer.py"),
        os.path.join(out_dir, "src"),
        os.path.join(out_dir, "models"),
    ):
        if os.path.isdir(legacy):
            shutil.rmtree(legacy)
        elif os.path.isfile(legacy):
            os.remove(legacy)

    with open(os.path.join(TEMPLATE_DIR, "transformer", "config.json")) as handle:
        transformer_config = json.load(handle)
    transformer_config.pop("auto_map", None)
    transformer_config.update({
        "_class_name": "GeoCoreTransformer2DModel",
        "model_size": model_size,
        "torch_dtype": dtype_name,
        "weights": weights,
        "training_steps": steps,
    })
    with open(os.path.join(out_dir, "transformer", "config.json"), "w") as handle:
        json.dump(transformer_config, handle, indent=2)
        handle.write("\n")

    with open(os.path.join(TEMPLATE_DIR, "scheduler", "scheduler_config.json")) as handle:
        scheduler_config = json.load(handle)
    scheduler_config.pop("auto_map", None)
    scheduler_config["_class_name"] = "GeoCoreFlowMatchEulerScheduler"
    with open(os.path.join(out_dir, "scheduler", "scheduler_config.json"), "w") as handle:
        json.dump(scheduler_config, handle, indent=2)
        handle.write("\n")

    with open(os.path.join(out_dir, "model_index.json"), "w") as handle:
        json.dump(MODEL_INDEX, handle, indent=2)
        handle.write("\n")


def sync_huggingface_templates() -> None:
    """Keep the in-repo `huggingface/` scaffold aligned with `src/diffusers`."""
    os.makedirs(os.path.join(TEMPLATE_DIR, "transformer"), exist_ok=True)
    os.makedirs(os.path.join(TEMPLATE_DIR, "scheduler"), exist_ok=True)
    shutil.copy2(
        os.path.join(SRC_DIFFUSERS, "pipelines", "geocore", "pipeline_geocore.py"),
        os.path.join(TEMPLATE_DIR, "pipeline.py"),
    )
    shutil.copy2(
        os.path.join(SRC_DIFFUSERS, "models", "transformers", "transformer_geocore.py"),
        os.path.join(TEMPLATE_DIR, "transformer", "transformer_geocore.py"),
    )
    shutil.copy2(
        os.path.join(SRC_DIFFUSERS, "schedulers", "scheduling_geocore.py"),
        os.path.join(TEMPLATE_DIR, "scheduler", "scheduling_geocore.py"),
    )
    for legacy in (
        os.path.join(TEMPLATE_DIR, "transformer", "modeling_geocore_transformer.py"),
        os.path.join(TEMPLATE_DIR, "bootstrap_geocore.py"),
    ):
        if os.path.isfile(legacy):
            os.remove(legacy)

    with open(os.path.join(TEMPLATE_DIR, "model_index.json"), "w") as handle:
        json.dump(MODEL_INDEX, handle, indent=2)
        handle.write("\n")

    with open(os.path.join(TEMPLATE_DIR, "transformer", "config.json")) as handle:
        transformer_config = json.load(handle)
    transformer_config.pop("auto_map", None)
    transformer_config["_class_name"] = "GeoCoreTransformer2DModel"
    with open(os.path.join(TEMPLATE_DIR, "transformer", "config.json"), "w") as handle:
        json.dump(transformer_config, handle, indent=2)
        handle.write("\n")

    with open(os.path.join(TEMPLATE_DIR, "scheduler", "scheduler_config.json")) as handle:
        scheduler_config = json.load(handle)
    scheduler_config.pop("auto_map", None)
    scheduler_config["_class_name"] = "GeoCoreFlowMatchEulerScheduler"
    with open(os.path.join(TEMPLATE_DIR, "scheduler", "scheduler_config.json"), "w") as handle:
        json.dump(scheduler_config, handle, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GeoCore-9B weights as a Diffusers model directory")
    parser.add_argument("--ckpt", default=None, help="Training checkpoint (.pt)")
    parser.add_argument("--out", default=None, help="Output directory")
    parser.add_argument("--weights", default="model", choices=["ema", "model"])
    parser.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    parser.add_argument("--model-size", default="9b", choices=list(MODEL_SIZES))
    parser.add_argument("--max-shard-size-gb", type=float, default=5.0)
    parser.add_argument(
        "--sync-templates-only",
        action="store_true",
        help="Only refresh huggingface/ remote-code templates from src/diffusers",
    )
    args = parser.parse_args()

    if args.sync_templates_only:
        sync_huggingface_templates()
        print(f"Synced Diffusers remote-code templates under {TEMPLATE_DIR}")
        return

    if not args.ckpt or not args.out:
        parser.error("--ckpt and --out are required unless --sync-templates-only is set")

    os.makedirs(args.out, exist_ok=True)
    dtype = DTYPES[args.dtype]

    print(f"Loading {args.ckpt} (mmap) ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)
    if args.weights not in ckpt:
        raise KeyError(f"'{args.weights}' not found in checkpoint; keys are {list(ckpt)}")

    state_dict = {
        strip_wrapper_prefixes(key): value.to(dtype).contiguous()
        for key, value in ckpt[args.weights].items()
    }
    steps = ckpt.get("steps")
    del ckpt

    dead = [key for key in ZERO_INIT_PROBES if key in state_dict and not state_dict[key].abs().any()]
    if dead:
        raise RuntimeError(f"'{args.weights}' state looks untrained: {dead} are still all-zero.")

    total_params = sum(value.numel() for value in state_dict.values())
    total_bytes = sum(value.numel() * value.element_size() for value in state_dict.values())
    print(f"{len(state_dict)} tensors | {total_params / 1e9:.2f}B params | {total_bytes / 1e9:.2f} GB {args.dtype}")

    write_transformer_shards(state_dict, args.out, int(args.max_shard_size_gb * 1e9))
    make_self_contained_repo(args.out, args.model_size, args.dtype, args.weights, steps)
    if os.path.abspath(args.out) == os.path.abspath(TEMPLATE_DIR):
        sync_huggingface_templates()
    print(f"Done. Wrote self-contained Diffusers layout to {args.out}")


if __name__ == "__main__":
    main()

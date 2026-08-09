"""Convert a GeoCore-9B training checkpoint into a Diffusers model directory.

    python scripts/convert_checkpoint.py \
        --ckpt exps/GeoCore-9B_new_rev/checkpoints/0300000.pt \
        --out huggingface/
"""
import argparse
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.flux2 import GeoCore4BParams, GeoCore9BParams

PARAMS = {"9b": GeoCore9BParams, "4b": GeoCore4BParams}
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
ZERO_INIT_PROBES = (
    "final_layer.linear.weight",
    "time_in.out_layer.weight",
    "double_stream_modulation_img.lin.weight",
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
TEMPLATE_DIR = os.path.join(REPO_ROOT, "huggingface")


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


def copy_geocore_diffusers_sources(out_dir: str) -> None:
    """Bundle in-tree Diffusers sources required for remote-code loading."""
    shutil.copy2(os.path.join(REPO_ROOT, "bootstrap_geocore.py"), os.path.join(out_dir, "bootstrap_geocore.py"))

    src_diffusers = os.path.join(REPO_ROOT, "src", "diffusers")
    dst_diffusers = os.path.join(out_dir, "src", "diffusers")
    if os.path.isdir(dst_diffusers):
        shutil.rmtree(dst_diffusers)
    shutil.copytree(src_diffusers, dst_diffusers)

    models_dir = os.path.join(out_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    for rel_path in ("__init__.py", "flux2.py"):
        shutil.copy2(os.path.join(REPO_ROOT, "models", rel_path), os.path.join(models_dir, rel_path))


def copy_diffusers_scaffold(out_dir: str, model_size: str, dtype_name: str, weights: str, steps: int | None) -> None:
    for rel_path in (
        "model_index.json",
        "pipeline.py",
        "scheduler/scheduler_config.json",
        "scheduler/scheduling_geocore.py",
        "transformer/modeling_geocore_transformer.py",
    ):
        src = os.path.join(TEMPLATE_DIR, rel_path)
        dst = os.path.join(out_dir, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    with open(os.path.join(TEMPLATE_DIR, "transformer", "config.json")) as handle:
        transformer_config = json.load(handle)
    transformer_config.update({
        "model_size": model_size,
        "torch_dtype": dtype_name,
        "weights": weights,
        "training_steps": steps,
    })
    with open(os.path.join(out_dir, "transformer", "config.json"), "w") as handle:
        json.dump(transformer_config, handle, indent=2)
    copy_geocore_diffusers_sources(out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GeoCore-9B weights as a Diffusers model directory")
    parser.add_argument("--ckpt", required=True, help="Training checkpoint (.pt)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--weights", default="model", choices=["ema", "model"])
    parser.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    parser.add_argument("--model-size", default="9b", choices=list(PARAMS))
    parser.add_argument("--max-shard-size-gb", type=float, default=5.0)
    args = parser.parse_args()

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
    copy_diffusers_scaffold(args.out, args.model_size, args.dtype, args.weights, steps)
    print(f"Done. Wrote Diffusers layout to {args.out}")


if __name__ == "__main__":
    main()

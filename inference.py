"""Text- and geo-conditioned sampling from GeoCore-9B via the Diffusers pipeline.

Example:
    python inference.py \
        --model-dir huggingface/ \
        --prompt "A satellite view of a highly dense urban city with towering skyscrapers" \
        --lon 126.97 --lat 37.56 --res 0.0 \
        --num-samples 4 --out samples/
"""
from __future__ import annotations

import argparse
import os

import torch

from bootstrap_geocore import ensure_geocore_diffusers

ensure_geocore_diffusers()

from geocore_diffusers import GeoCorePipeline

torch.set_float32_matmul_precision("high")

NULL_META = -999.0


def load_pipeline(model_dir: str, dtype: torch.dtype, lora: str | None = None):
    pipe = GeoCorePipeline.from_pretrained(
        model_dir,
        torch_dtype=dtype,
    )
    if lora:
        from peft import PeftModel
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora).merge_and_unload()
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description="GeoCore-9B text-to-image sampling")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Diffusers model directory (exported or Hub snapshot)")
    parser.add_argument("--lora", type=str, default=None, help="Optional LoRA adapter directory to merge")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--res", type=float, default=NULL_META,
                        help="Resolution index (17 - tile zoom). Unset means null.")
    parser.add_argument("--lon", type=float, default=NULL_META, help="Longitude in degrees; unset means null")
    parser.add_argument("--lat", type=float, default=NULL_META, help="Latitude in degrees; unset means null")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--out", type=str, default="samples")
    args = parser.parse_args()

    if args.resolution % 16 != 0:
        raise ValueError("Resolution must be divisible by 16.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]

    os.makedirs(args.out, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    pipe = load_pipeline(args.model_dir, weight_dtype, args.lora).to(device)

    lon, lat = args.lon, args.lat
    if lon == NULL_META or lat == NULL_META:
        lon = lat = NULL_META

    written = 0
    while written < args.num_samples:
        batch = min(args.batch_size, args.num_samples - written)
        result = pipe(
            prompt=[args.prompt] * batch,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.num_steps,
            guidance_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            res=args.res,
            lon=lon,
            lat=lat,
            generator=generator,
            output_type="pil",
        )
        for image in result.images:
            image.save(os.path.join(args.out, f"sample_{written:04d}.png"))
            written += 1

    print(f"Saved {written} sample(s) to {args.out}")


if __name__ == "__main__":
    main()

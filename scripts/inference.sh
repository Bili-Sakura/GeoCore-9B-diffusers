#!/usr/bin/env bash
# Sample from GeoCore-9B with text + geospatial conditioning via the Diffusers pipeline.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:?"set MODEL_DIR to a Diffusers model directory (exported or Hub snapshot)"}

python inference.py \
    --model-dir "$MODEL_DIR" \
    --prompt "A satellite view of a highly dense urban city with towering skyscrapers" \
    --lon 126.97 --lat 37.56 --res 0.0 \
    --num-samples 4 \
    --num-steps 50 \
    --cfg-scale 4.0 \
    --out samples \
    "$@"

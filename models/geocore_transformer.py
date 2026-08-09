# Copyright 2026 Jeonghyeok Do and Munchurl Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusers-compatible wrapper around the GeoCore Flow Matching DiT."""

from __future__ import annotations

from dataclasses import asdict, fields

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from models.flux2 import Flux2, GeoCore4BParams, GeoCore9BParams

PARAMS = {"9b": GeoCore9BParams, "4b": GeoCore4BParams}


class GeoCoreTransformer2DModel(Flux2, ModelMixin, ConfigMixin):
    r"""GeoCore text- and geo-conditioned Flow Matching DiT."""

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        context_in_dim: int = 4096,
        hidden_size: int = 4096,
        num_heads: int = 32,
        depth: int = 8,
        depth_repa: int = 8,
        depth_single_blocks: int = 24,
        axes_dim: list[int] | None = None,
        theta: int = 2000,
        mlp_ratio: float = 3.0,
        y_in_dim: int = 768,
        z_out_dim: int = 4096,
        model_size: str = "9b",
        sample_size: int = 16,
        patch_size: int = 2,
        latent_channels: int = 128,
        **_unused,
    ) -> None:
        ModelMixin.__init__(self)
        params_cls = PARAMS[model_size]
        arch_fields = {field.name for field in fields(params_cls)}
        arch_kwargs = {
            key: value
            for key, value in {
                "in_channels": in_channels,
                "context_in_dim": context_in_dim,
                "hidden_size": hidden_size,
                "num_heads": num_heads,
                "depth": depth,
                "depth_repa": depth_repa,
                "depth_single_blocks": depth_single_blocks,
                "axes_dim": axes_dim or [32, 48, 48],
                "theta": theta,
                "mlp_ratio": mlp_ratio,
                "y_in_dim": y_in_dim,
                "z_out_dim": z_out_dim,
            }.items()
            if key in arch_fields
        }
        Flux2.__init__(self, params_cls(**arch_kwargs))

    @classmethod
    def from_flux2_params(cls, params: GeoCore9BParams | GeoCore4BParams, model_size: str) -> "GeoCoreTransformer2DModel":
        return cls(model_size=model_size, **asdict(params))

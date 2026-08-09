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
"""Diffusers pipeline for GeoCore text- and geo-conditioned EO image generation."""

from __future__ import annotations

from typing import Any

import torch
from diffusers import AutoencoderKLFlux2
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from geocore_diffusers.models.transformers.transformer_geocore import GeoCoreTransformer2DModel
from geocore_diffusers.pipelines.geocore.pipeline_output import GeoCorePipelineOutput
from geocore_diffusers.schedulers.scheduling_geocore import GeoCoreFlowMatchEulerScheduler

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

NULL_META = -999.0
DEFAULT_CLIP_REPO = "openai/clip-vit-large-patch14"
DEFAULT_T5_REPO = "google/t5-v1_1-xxl"
TEXT_SEQ_LEN = 256

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> from pathlib import Path
        >>> import torch
        >>> from geocore_diffusers import GeoCorePipeline

        >>> model_dir = Path("/path/to/GeoCore-9B")
        >>> pipe = GeoCorePipeline.from_pretrained(
        ...     str(model_dir),
        ...     torch_dtype=torch.bfloat16,
        ... )
        >>> pipe = pipe.to("cuda")

        >>> image = pipe(
        ...     prompt="A satellite view of a highly dense urban city with towering skyscrapers",
        ...     lon=126.97,
        ...     lat=37.56,
        ...     res=0.0,
        ...     num_inference_steps=50,
        ...     guidance_scale=4.0,
        ...     generator=torch.Generator("cuda").manual_seed(0),
        ... ).images[0]
        ```
"""


class GeoCorePipeline(DiffusionPipeline):
    r"""Text- and geospatial-metadata-conditioned sampling for GeoCore-9B."""

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]

    def __init__(
        self,
        transformer: GeoCoreTransformer2DModel,
        vae: AutoencoderKLFlux2,
        scheduler: GeoCoreFlowMatchEulerScheduler,
        text_encoder: CLIPTextModel | None = None,
        text_encoder_2: T5EncoderModel | None = None,
        tokenizer: CLIPTokenizer | None = None,
        tokenizer_2: T5TokenizerFast | None = None,
    ) -> None:
        super().__init__()

        if text_encoder is None or tokenizer is None:
            tokenizer = tokenizer or CLIPTokenizer.from_pretrained(DEFAULT_CLIP_REPO)
            text_encoder = text_encoder or CLIPTextModel.from_pretrained(DEFAULT_CLIP_REPO)
        if text_encoder_2 is None or tokenizer_2 is None:
            tokenizer_2 = tokenizer_2 or T5TokenizerFast.from_pretrained(DEFAULT_T5_REPO)
            text_encoder_2 = text_encoder_2 or T5EncoderModel.from_pretrained(DEFAULT_T5_REPO)

        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self._null_pooled: torch.Tensor | None = None
        self._null_prompt: torch.Tensor | None = None

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1.0

    def _cache_null_text_embeddings(self) -> None:
        if self._null_prompt is not None and self._null_pooled is not None:
            return

        device = self._execution_device
        dtype = self.text_encoder.dtype
        clip_inputs = self.tokenizer(
            [""], padding="max_length", max_length=77, truncation=True, return_tensors="pt",
        ).to(device)
        self._null_pooled = self.text_encoder(clip_inputs.input_ids).pooler_output.to(dtype=dtype)
        t5_inputs = self.tokenizer_2(
            [""], padding="max_length", max_length=TEXT_SEQ_LEN, truncation=True, return_tensors="pt",
        ).to(device)
        self._null_prompt = self.text_encoder_2(t5_inputs.input_ids)[0].to(dtype=dtype)

    def check_inputs(
        self,
        prompt: str | list[str],
        height: int,
        width: int,
        guidance_scale: float,
        lon: float | list[float] | None,
        lat: float | list[float] | None,
        res: float | list[float] | None,
    ) -> None:
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 16, got {height}x{width}.")
        if guidance_scale < 0:
            raise ValueError(f"`guidance_scale` must be >= 0, got {guidance_scale}.")
        if isinstance(prompt, list) and len(prompt) == 0:
            raise ValueError("`prompt` must be a non-empty string or list of strings.")

    @staticmethod
    def _prepare_latent_image_ids(height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height, device=device, dtype=dtype)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width, device=device, dtype=dtype)[None, :]
        return latent_image_ids.reshape(height * width, 3)

    def _encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device,
        dtype: torch.dtype,
        num_images_per_prompt: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt)
        prompt_embeds = torch.empty((batch_size, TEXT_SEQ_LEN, 4096), device=device, dtype=dtype)
        pooled_embeds = torch.empty((batch_size, 768), device=device, dtype=dtype)
        text_ids = torch.zeros((batch_size, TEXT_SEQ_LEN, 3), device=device, dtype=dtype)
        text_ids[:, :, 0] = torch.arange(TEXT_SEQ_LEN, device=device, dtype=dtype)

        self._cache_null_text_embeddings()
        assert self._null_prompt is not None and self._null_pooled is not None

        empty_indices = [index for index, text in enumerate(prompt) if text == ""]
        valid_indices = [index for index, text in enumerate(prompt) if text != ""]

        if empty_indices:
            prompt_embeds[empty_indices] = self._null_prompt.expand(len(empty_indices), -1, -1)
            pooled_embeds[empty_indices] = self._null_pooled.expand(len(empty_indices), -1)

        if valid_indices:
            valid_prompts = [prompt[index] for index in valid_indices]
            clip_inputs = self.tokenizer(
                valid_prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt",
            ).to(device)
            pooled_embeds[valid_indices] = self.text_encoder(clip_inputs.input_ids).pooler_output
            t5_inputs = self.tokenizer_2(
                valid_prompts, padding="max_length", max_length=TEXT_SEQ_LEN, truncation=True, return_tensors="pt",
            ).to(device)
            prompt_embeds[valid_indices] = self.text_encoder_2(t5_inputs.input_ids)[0]

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            pooled_embeds = pooled_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            text_ids = text_ids.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, pooled_embeds, text_ids

    def _get_null_conditioning(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        self._cache_null_text_embeddings()
        assert self._null_prompt is not None and self._null_pooled is not None
        return {
            "ctx": self._null_prompt.expand(batch_size, -1, -1).to(device=device, dtype=dtype),
            "y": self._null_pooled.expand(batch_size, -1).to(device=device, dtype=dtype),
            "res": torch.zeros(batch_size, device=device, dtype=dtype),
            "lon": torch.zeros(batch_size, device=device, dtype=dtype),
            "lat": torch.zeros(batch_size, device=device, dtype=dtype),
        }

    @staticmethod
    def _expand_batch(
        value: float | list[float] | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        default: float = NULL_META,
    ) -> torch.Tensor:
        if value is None:
            return torch.full((batch_size,), default, device=device, dtype=dtype)
        if isinstance(value, (int, float)):
            return torch.full((batch_size,), float(value), device=device, dtype=dtype)
        if len(value) == 1:
            return torch.full((batch_size,), float(value[0]), device=device, dtype=dtype)
        return torch.tensor(value, device=device, dtype=dtype)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        shape = (batch_size, num_channels, height // self.vae_scale_factor, width // self.vae_scale_factor)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def _predict_velocity(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        cond_kwargs: dict[str, torch.Tensor],
        uncond_kwargs: dict[str, torch.Tensor],
        guidance_scale: float,
        guidance_low: float,
        guidance_high: float,
    ) -> torch.Tensor:
        t_value = float(timestep[0].item()) if timestep.ndim > 0 else float(timestep.item())
        if guidance_scale > 1.0 and guidance_low <= t_value <= guidance_high:
            model_input = torch.cat([hidden_states] * 2, dim=0)
            t_input = torch.cat([timestep] * 2, dim=0)
            combined_kwargs: dict[str, torch.Tensor] = {}
            for key, value in cond_kwargs.items():
                if key in ("ctx", "y", "res", "lon", "lat"):
                    uncond_value = uncond_kwargs.get(key, value)
                    combined_kwargs[key] = torch.cat([value, uncond_value], dim=0)
                elif key in ("x_ids", "ctx_ids"):
                    if value.dim() == 3 and value.shape[0] == hidden_states.shape[0]:
                        combined_kwargs[key] = torch.cat([value] * 2, dim=0)
                    else:
                        combined_kwargs[key] = value
                else:
                    combined_kwargs[key] = value
            x_ids = combined_kwargs.pop("x_ids")
            velocity = self.transformer(model_input, x_ids, t_input, **combined_kwargs)
            if isinstance(velocity, tuple):
                velocity = velocity[0]
            velocity_cond, velocity_uncond = velocity.chunk(2)
            return velocity_uncond + guidance_scale * (velocity_cond - velocity_uncond)

        x_ids = cond_kwargs["x_ids"]
        velocity = self.transformer(
            hidden_states, x_ids, timestep, **{k: v for k, v in cond_kwargs.items() if k != "x_ids"},
        )
        if isinstance(velocity, tuple):
            velocity = velocity[0]
        return velocity

    @replace_example_docstring(EXAMPLE_DOC_STRING)
    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str],
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        guidance_low: float = 0.0,
        guidance_high: float = 1.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        lon: float | list[float] | None = None,
        lat: float | list[float] | None = None,
        res: float | list[float] | None = None,
        callback_on_step_end: Any | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
    ) -> GeoCorePipelineOutput | tuple:
        r"""
        Generate Earth-observation images conditioned on text and geospatial metadata.

        Examples:
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        self.check_inputs(prompt, height, width, guidance_scale, lon, lat, res)
        self._guidance_scale = guidance_scale

        device = self._execution_device
        dtype = self.transformer.dtype

        if lon is None or lat is None or (isinstance(lon, float) and lon == NULL_META) or (isinstance(lat, float) and lat == NULL_META):
            lon = lat = NULL_META
        elif isinstance(lon, list) or isinstance(lat, list):
            lon_list = lon if isinstance(lon, list) else [lon] * len(prompt)
            lat_list = lat if isinstance(lat, list) else [lat] * len(prompt)
            lon = [NULL_META if l == NULL_META or la == NULL_META else l for l, la in zip(lon_list, lat_list)]
            lat = [NULL_META if l == NULL_META or la == NULL_META else la for l, la in zip(lon_list, lat_list)]

        batch_size = len(prompt) * num_images_per_prompt
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        latent_channels = self.transformer.config.latent_channels

        prompt_embeds, pooled_embeds, text_ids = self._encode_prompt(
            prompt, device=device, dtype=dtype, num_images_per_prompt=num_images_per_prompt,
        )

        cond_kwargs = {
            "ctx": prompt_embeds,
            "ctx_ids": text_ids,
            "y": pooled_embeds,
            "res": self._expand_batch(res, batch_size, device, dtype),
            "lon": self._expand_batch(lon, batch_size, device, dtype),
            "lat": self._expand_batch(lat, batch_size, device, dtype),
            "x_ids": self._prepare_latent_image_ids(latent_height, latent_width, device, dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1),
        }
        uncond_kwargs = self._get_null_conditioning(batch_size, device, dtype)
        uncond_kwargs["ctx_ids"] = text_ids

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        latents = self.prepare_latents(
            batch_size, latent_channels, height, width, dtype, device, generator, latents,
        )

        for step_index, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            timestep = t.expand(latents.shape[0]).to(device=device, dtype=dtype)
            velocity = self._predict_velocity(
                latents, timestep, cond_kwargs, uncond_kwargs, guidance_scale, guidance_low, guidance_high,
            )
            latents = self.scheduler.step(velocity, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {"prompt": prompt, "latents": latents, "step_index": step_index}
                callback_outputs = callback_on_step_end(self, step_index, timestep, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)

            if XLA_AVAILABLE:
                xm.mark_step()

        if output_type == "latent":
            if not return_dict:
                return (latents,)
            return GeoCorePipelineOutput(images=latents)

        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return GeoCorePipelineOutput(images=image)

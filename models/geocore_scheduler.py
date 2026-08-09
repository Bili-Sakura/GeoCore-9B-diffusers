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
"""Flow-matching Euler scheduler for GeoCore (continuous t in [0, 1])."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput


@dataclass
class GeoCoreFlowMatchEulerSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class GeoCoreFlowMatchEulerScheduler(SchedulerMixin, ConfigMixin):
    r"""Euler integrator for rectified-flow sampling with timesteps in ``[0, 1]``."""

    order = 1

    @register_to_config
    def __init__(self, num_train_timesteps: int = 1000) -> None:
        self.timesteps: torch.Tensor | None = None
        self.next_timesteps: torch.Tensor | None = None
        self.num_inference_steps: int | None = None
        self._step_index: int | None = None

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device | None = None) -> None:
        schedule = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float64)
        self.timesteps = schedule[:-1]
        self.next_timesteps = schedule[1:]
        self.num_inference_steps = num_inference_steps
        self._step_index = 0

    def scale_model_input(self, sample: torch.Tensor, timestep: int | torch.Tensor) -> torch.Tensor:
        return sample

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int | torch.Tensor,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> GeoCoreFlowMatchEulerSchedulerOutput | tuple[torch.Tensor]:
        if self._step_index is None or self.timesteps is None or self.next_timesteps is None:
            raise ValueError("Call `set_timesteps` before `step`.")

        t_cur = self.timesteps[self._step_index]
        t_next = self.next_timesteps[self._step_index]
        prev_sample = sample + (t_next - t_cur) * model_output
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return GeoCoreFlowMatchEulerSchedulerOutput(prev_sample=prev_sample)

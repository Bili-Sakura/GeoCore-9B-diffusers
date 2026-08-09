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
"""Remote-code entry point for GeoCore transformer weights."""

from bootstrap_geocore import add_repo_root_for, ensure_geocore_diffusers

add_repo_root_for(__file__)
ensure_geocore_diffusers()

from geocore_diffusers.models.transformers.transformer_geocore import GeoCoreTransformer2DModel

__all__ = ["GeoCoreTransformer2DModel"]

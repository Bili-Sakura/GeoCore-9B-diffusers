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
"""Bootstrap `geocore_diffusers` from a source checkout without pip install.

Hub / Diffusers custom-pipeline inference does **not** need this module: exported
model directories embed self-contained remote-code copies under `pipeline.py`,
`transformer/`, and `scheduler/`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def ensure_geocore_diffusers() -> None:
    """Make `geocore_diffusers` importable from the in-tree `src/diffusers` package."""
    try:
        import geocore_diffusers  # noqa: F401
        return
    except ImportError:
        pass

    repo_root = Path(__file__).resolve().parent
    package_root = repo_root / "src" / "diffusers"
    if not package_root.is_dir():
        raise ImportError(f"GeoCore Diffusers sources not found at {package_root}")

    spec = importlib.util.spec_from_file_location(
        "geocore_diffusers",
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Failed to load geocore_diffusers from source tree")

    module = importlib.util.module_from_spec(spec)
    sys.modules["geocore_diffusers"] = module
    spec.loader.exec_module(module)

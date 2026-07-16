# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from PIL import Image

from reasyn.chem.mol import Molecule


@dataclass(frozen=True)
class ReactionStep:
    rxn_id: int
    smarts: str
    reactants: tuple[Molecule, ...]
    product: Molecule | None


@dataclass
class DrawOptions:
    width: int = 900
    panel_height: int = 360
    max_panel_height: int = 520


@dataclass
class StepPage:
    step_idx: int
    n_steps: int
    rxn_id: int
    caption_lines: list[str]
    reactant_smiles: list[str]
    product_smiles: str | None
    smarts: str
    image: Image.Image | None
    warning: str | None = None


@dataclass
class RouteDocument:
    route_index: int
    target: str
    product: str
    score: float
    num_steps: int
    synthesis: str
    model: str | None = None
    overview_image: Image.Image | None = None
    steps: list[StepPage] = field(default_factory=list)

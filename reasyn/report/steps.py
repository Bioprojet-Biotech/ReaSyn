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

from __future__ import annotations

import pickle
import re
from pathlib import Path

from reasyn.chem.matrix import ReactantReactionMatrix
from reasyn.chem.mol import Molecule
from reasyn.chem.stack import Stack
from reasyn.report.types import ReactionStep

RXN_TOKEN = re.compile(r"^R(\d+)$")


def load_rxn_matrix(matrix_path: Path | str) -> ReactantReactionMatrix:
    path = Path(matrix_path)
    with path.open("rb") as handle:
        return pickle.load(handle)


def _stack_reactants(stack: Stack, num_reactants: int) -> tuple[Molecule, ...]:
    if num_reactants == 1:
        return (stack.get_one_top(),)
    if num_reactants == 2:
        return (stack.get_one_top(), next(iter(stack.get_second_top())))
    if num_reactants == 3:
        return (
            stack.get_one_top(),
            next(iter(stack.get_second_top())),
            next(iter(stack.get_third_top())),
        )
    raise ValueError(f"Unsupported reactant count: {num_reactants}")


def pathway_to_steps(
    synthesis: str,
    rxn_matrix: ReactantReactionMatrix,
    final_smiles: str = "",
) -> list[ReactionStep]:
    del final_smiles  # kept for API compatibility; stack replay defines intermediates.

    stack = Stack()
    steps: list[ReactionStep] = []

    for token in synthesis.split(";"):
        token = token.strip()
        if not token:
            continue

        match = RXN_TOKEN.match(token)
        if match:
            rxn_id = int(match.group(1))
            rxn = rxn_matrix.reactions[rxn_id]
            reactants = _stack_reactants(stack, rxn.num_reactants)
            if not stack.push_rxn(rxn, rxn_id):
                steps.append(
                    ReactionStep(
                        rxn_id=rxn_id,
                        smarts=rxn.smarts,
                        reactants=reactants,
                        product=None,
                    )
                )
                continue

            steps.append(
                ReactionStep(
                    rxn_id=rxn_id,
                    smarts=rxn.smarts,
                    reactants=reactants,
                    product=stack.mols[-1],
                )
            )
        else:
            stack.push_mol(Molecule(token), 0)

    return steps

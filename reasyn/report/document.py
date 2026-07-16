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

import warnings

import pandas as pd

from reasyn.chem.matrix import ReactantReactionMatrix
from reasyn.report.draw import (
    annotate_image,
    draw_overview,
    draw_reaction_step,
    placeholder_image,
    reaction_step_caption,
)
from reasyn.report.steps import pathway_to_steps
from reasyn.report.types import DrawOptions, RouteDocument, StepPage


def build_route(
    row: pd.Series,
    rxn_matrix: ReactantReactionMatrix,
    opts: DrawOptions,
    route_index: int = 0,
) -> RouteDocument:
    wrap_chars = max(40, opts.width // 8)
    overview = draw_overview(
        target=str(row["target"]),
        product=str(row["smiles"]),
        score=float(row["score"]),
        num_steps=int(row["num_steps"]),
        opts=opts,
    )

    steps = pathway_to_steps(str(row["synthesis"]), rxn_matrix, str(row["smiles"]))
    n_steps = len(steps)
    step_pages: list[StepPage] = []

    for step_idx, step in enumerate(steps, start=1):
        caption = reaction_step_caption(step_idx, step, wrap_chars)
        warning: str | None = None
        image = None
        try:
            reaction_image = draw_reaction_step(step, opts)
            image = annotate_image(reaction_image, caption, width=opts.width)
        except Exception as exc:  # noqa: BLE001 — keep report generation resilient
            warning = f"Failed to draw step {step_idx} (R{step.rxn_id}): {exc}"
            warnings.warn(warning, stacklevel=2)
            image = annotate_image(
                placeholder_image(opts.width, opts.panel_height, warning),
                caption,
                width=opts.width,
            )

        step_pages.append(
            StepPage(
                step_idx=step_idx,
                n_steps=n_steps,
                rxn_id=step.rxn_id,
                caption_lines=caption,
                reactant_smiles=[m.smiles for m in step.reactants],
                product_smiles=step.product.smiles if step.product is not None else None,
                smarts=step.smarts,
                image=image,
                warning=warning,
            )
        )

    model = str(row["model"]) if "model" in row.index and pd.notna(row.get("model")) else None

    return RouteDocument(
        route_index=route_index,
        target=str(row["target"]),
        product=str(row["smiles"]),
        score=float(row["score"]),
        num_steps=int(row["num_steps"]),
        synthesis=str(row["synthesis"]),
        model=model,
        overview_image=overview,
        steps=step_pages,
    )


def build_routes(
    df: pd.DataFrame,
    rxn_matrix: ReactantReactionMatrix,
    opts: DrawOptions,
) -> list[RouteDocument]:
    return [
        build_route(row, rxn_matrix, opts, route_index=i)
        for i, (_, row) in enumerate(df.iterrows())
    ]

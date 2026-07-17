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

import re
from pathlib import Path
from typing import Literal

from reasyn.report.document import build_routes
from reasyn.report.io import load_results_table
from reasyn.report.pdf import write_pdf
from reasyn.report.select import select_routes
from reasyn.report.steps import load_rxn_matrix
from reasyn.report.types import DrawOptions

_TOP_RE = re.compile(r"^top\((\d+)\)$")

ReportMode = Literal["all"] | int


def parse_report_arg(value: str | None) -> ReportMode | None:
    """Parse ``--report``: ``None``, ``\"all\"``, or ``top(n)`` → positive int."""
    if value is None:
        return None
    if value == "all":
        return "all"
    match = _TOP_RE.fullmatch(value)
    if match is not None:
        n = int(match.group(1))
        if n < 1:
            raise ValueError('n in "top(n)" must be a positive integer')
        return n
    raise ValueError('Invalid --report value; use "all" or "top(n)" (e.g. top(1))')


def write_report_from_arg(results: Path, report: str | None) -> tuple[Path, int] | None:
    """Parse ``report`` and write a PDF next to ``results`` (stem + ``.pdf``).

    Returns ``(pdf_path, num_routes)``, or ``None`` if ``report`` is ``None``.
    """
    mode = parse_report_arg(report)
    if mode is None:
        return None
    if mode == "all":
        return generate_pathways_report(results, all_routes=True)
    return generate_pathways_report(results, top_k=mode)


def generate_pathways_report(
    results: Path,
    *,
    output: Path | None = None,
    all_routes: bool = False,
    top_k: int = 1,
    matrix_path: Path = Path("data/processed/comp_2048/matrix.pkl"),
    indices: list[int] | None = None,
    min_score: float | None = None,
    width: int = 900,
    panel_height: int = 360,
    max_panel_height: int = 520,
) -> tuple[Path, int]:
    """Build a PDF pathway report from a sampling results table.

    If ``output`` is ``None``, writes to ``results`` with the suffix replaced by
    ``.pdf``.

    Returns ``(pdf_path, num_routes)``.
    """
    if output is None:
        output = results.with_suffix(".pdf")
    elif output.suffix.lower() != ".pdf":
        raise ValueError("Output must be a .pdf file")

    df = load_results_table(results)
    selected = select_routes(
        df,
        indices=indices,
        top_k=None if all_routes else top_k,
        min_score=min_score,
    )
    if selected.empty:
        raise ValueError("No routes selected for the report")

    opts = DrawOptions(width=width, panel_height=panel_height, max_panel_height=max_panel_height)
    rxn_matrix = load_rxn_matrix(matrix_path)
    routes = build_routes(selected, rxn_matrix, opts)
    return write_pdf(routes, output), len(routes)

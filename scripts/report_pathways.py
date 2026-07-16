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

import sys
sys.path.append(".")

from pathlib import Path

import click

from reasyn.report import (
    DrawOptions,
    build_routes,
    load_results_table,
    load_rxn_matrix,
    select_routes,
    write_pdf,
)


def _parse_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return None
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise click.BadParameter("indices must be comma-separated integers") from exc


@click.command()
@click.argument("results", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), required=True)
@click.option(
    "--matrix-path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/processed/comp_2048/matrix.pkl"),
)
@click.option(
    "--indices",
    type=str,
    default=None,
    help="Comma-separated 0-based row indices (applied after --min-score if set).",
)
@click.option("--top-k", type=int, default=1, show_default=True, help="Best K pathways per target.")
@click.option("--all", "all_routes", is_flag=True, help="Include all routes (ignores --top-k).")
@click.option("--min-score", type=float, default=None, help="Optional minimum Tanimoto score.")
@click.option("--width", type=int, default=900, show_default=True)
@click.option("--panel-height", type=int, default=360, show_default=True)
@click.option("--max-panel-height", type=int, default=520, show_default=True)
def main(
    results: Path,
    output: Path,
    matrix_path: Path,
    indices: str | None,
    top_k: int,
    all_routes: bool,
    min_score: float | None,
    width: int,
    panel_height: int,
    max_panel_height: int,
) -> None:
    """Build a Reaxys-style PDF report (overview + one page per step) from sampling results."""
    if output.suffix.lower() != ".pdf":
        raise click.ClickException("Output must be a .pdf file")

    try:
        df = load_results_table(results)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    idx = _parse_indices(indices)
    try:
        selected = select_routes(
            df,
            indices=idx,
            top_k=None if all_routes else top_k,
            min_score=min_score,
        )
    except (IndexError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if selected.empty:
        raise click.ClickException("No routes selected for the report")

    opts = DrawOptions(width=width, panel_height=panel_height, max_panel_height=max_panel_height)
    rxn_matrix = load_rxn_matrix(matrix_path)
    routes = build_routes(selected, rxn_matrix, opts)
    path = write_pdf(routes, output)
    click.echo(f"Wrote {path} ({len(routes)} route(s))")


if __name__ == "__main__":
    main()

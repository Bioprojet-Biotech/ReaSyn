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

from reasyn.report import DrawOptions, load_results_table, load_rxn_matrix, select_routes, write_html_report


@click.command()
@click.argument("results", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=Path("results/rendered"))
@click.option(
    "--matrix-path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/processed/comp_2048/matrix.pkl"),
)
@click.option("--top-k", type=int, default=1, show_default=True, help="Best K pathways per target.")
@click.option("--min-score", type=float, default=None, help="Optional minimum score filter.")
@click.option("--width", type=int, default=900, show_default=True)
@click.option(
    "--panel-height",
    type=int,
    default=360,
    show_default=True,
    help="Height of each molecule/reaction panel.",
)
@click.option(
    "--max-panel-height",
    type=int,
    default=520,
    show_default=True,
    help="Max auto-grown panel height for complex steps.",
)
def main(
    results: Path,
    output: Path,
    matrix_path: Path,
    top_k: int,
    min_score: float | None,
    width: int,
    panel_height: int,
    max_panel_height: int,
) -> None:
    """Render ReaSyn results as reaction diagrams (PNG + HTML index)."""
    try:
        df = load_results_table(results)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        selected = select_routes(df, top_k=top_k, min_score=min_score)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if selected.empty:
        raise click.ClickException("No routes selected to render")

    opts = DrawOptions(width=width, panel_height=panel_height, max_panel_height=max_panel_height)
    rxn_matrix = load_rxn_matrix(matrix_path)
    index_path = write_html_report(selected, rxn_matrix, output, opts)
    click.echo(f"Wrote {index_path}")


if __name__ == "__main__":
    main()

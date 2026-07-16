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

import hashlib
import html
from pathlib import Path

import pandas as pd

from reasyn.chem.matrix import ReactantReactionMatrix
from reasyn.report.document import build_route
from reasyn.report.draw import stack_images
from reasyn.report.types import DrawOptions, RouteDocument


def _safe_stem(smiles: str, index: int) -> str:
    digest = hashlib.md5(smiles.encode(), usedforsecurity=False).hexdigest()[:10]
    return f"pathway_{index:03d}_{digest}"


def render_route_png(
    route: RouteDocument,
    output_dir: Path,
    width: int,
) -> tuple[Path, list[str]]:
    images = []
    if route.overview_image is not None:
        images.append(route.overview_image)
    step_summaries: list[str] = []
    for step in route.steps:
        if step.image is not None:
            images.append(step.image)
        step_summaries.append(" ".join(step.caption_lines))

    if not images:
        raise ValueError(f"Route {route.route_index} has no images to render")

    combined = stack_images(images, width=width)
    output_path = output_dir / f"{_safe_stem(route.target, route.route_index)}.png"
    combined.save(output_path)
    return output_path, step_summaries


def write_html_index(rows: list[dict], output_dir: Path) -> Path:
    index_path = output_dir / "index.html"
    cards = []
    for item in rows:
        steps_html = ""
        if item.get("step_details"):
            steps_html = "<h3>Reaction steps</h3><ol>" + "".join(
                f"<li><code>{html.escape(step)}</code></li>" for step in item["step_details"]
            ) + "</ol>"
        cards.append(
            "<section>"
            f"<h2>Target</h2><p><code>{html.escape(item['target'])}</code></p>"
            f"<h3>Product (score={html.escape(item['score'])})</h3>"
            f"<p><code>{html.escape(item['product'])}</code></p>"
            f"<p>Steps: {html.escape(item['num_steps'])}</p>"
            f"{steps_html}"
            f'<img src="{html.escape(item["image"])}" alt="pathway" />'
            f"<details><summary>Synthesis string</summary>"
            f"<pre>{html.escape(item['synthesis'])}</pre></details>"
            "</section>"
        )

    index_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>ReaSyn pathway render</title>"
        "<style>"
        "body{font-family:sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;}"
        "section{border:1px solid #ddd;border-radius:8px;padding:1rem;margin-bottom:1.5rem;}"
        "img{max-width:100%;height:auto;border:1px solid #eee;}"
        "code,pre{word-break:break-all;white-space:pre-wrap;}"
        "</style></head><body>"
        "<h1>ReaSyn pathway renders</h1>"
        + "\n".join(cards)
        + "</body></html>",
        encoding="utf-8",
    )
    return index_path


def write_html_report(
    df: pd.DataFrame,
    rxn_matrix: ReactantReactionMatrix,
    output_dir: Path,
    opts: DrawOptions,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    html_rows: list[dict] = []
    for i, (_, row) in enumerate(df.iterrows()):
        route = build_route(row, rxn_matrix, opts, route_index=i)
        image_path, step_summaries = render_route_png(route, output_dir, opts.width)
        html_rows.append(
            {
                "target": route.target,
                "product": route.product,
                "score": f"{route.score:.3f}",
                "num_steps": str(route.num_steps),
                "synthesis": route.synthesis,
                "image": image_path.name,
                "step_details": step_summaries,
            }
        )
    return write_html_index(html_rows, output_dir)

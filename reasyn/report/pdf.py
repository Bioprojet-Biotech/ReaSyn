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

from io import BytesIO
from pathlib import Path

from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from reasyn.report.types import RouteDocument


def _pil_to_flowable(image: Image.Image, max_width: float, max_height: float) -> RLImage:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    w, h = image.size
    scale = min(max_width / w, max_height / h, 1.0)
    return RLImage(buf, width=w * scale, height=h * scale)


def _truncate(text: str, max_len: int = 240) -> str:
    text = text.replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def write_pdf(routes: list[RouteDocument], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    page_w, page_h = A4
    margin = 0.6 * inch
    usable_w = page_w - 2 * margin
    usable_h = page_h - 2 * margin

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "RouteTitle",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "StepHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "BodyMono",
        parent=styles["Normal"],
        fontSize=8,
        leading=11,
        wordWrap="CJK",
    )

    story: list = []

    if len(routes) > 1:
        story.append(Paragraph("ReaSyn pathway report — index", title_style))
        for route in routes:
            story.append(
                Paragraph(
                    f"Route {route.route_index}: score={route.score:.3f}, "
                    f"steps={route.num_steps}, product=<font face='Courier'>"
                    f"{_truncate(route.product, 80)}</font>",
                    body_style,
                )
            )
        story.append(PageBreak())

    for route_i, route in enumerate(routes):
        story.append(Paragraph(f"Route {route.route_index} — overview", title_style))
        story.append(
            Paragraph(
                f"Score: {route.score:.3f} &nbsp;&nbsp; Steps: {route.num_steps}",
                body_style,
            )
        )
        if route.model:
            story.append(Paragraph(f"Model: {route.model}", body_style))
        story.append(Paragraph(f"Target: <font face='Courier'>{route.target}</font>", body_style))
        story.append(Paragraph(f"Product: <font face='Courier'>{route.product}</font>", body_style))
        story.append(
            Paragraph(
                f"Synthesis: <font face='Courier'>{_truncate(route.synthesis, 400)}</font>",
                body_style,
            )
        )
        story.append(Spacer(1, 0.2 * inch))
        if route.overview_image is not None:
            story.append(
                _pil_to_flowable(route.overview_image, usable_w, usable_h * 0.55)
            )

        for step in route.steps:
            story.append(PageBreak())
            story.append(
                Paragraph(
                    f"Route {route.route_index} — Step {step.step_idx} / {step.n_steps} · R{step.rxn_id}",
                    heading_style,
                )
            )
            if step.warning:
                story.append(Paragraph(f"Warning: {step.warning}", body_style))
            story.append(
                Paragraph(
                    "Reactants: "
                    + ", ".join(f"<font face='Courier'>{s}</font>" for s in step.reactant_smiles),
                    body_style,
                )
            )
            prod = step.product_smiles or "(failed)"
            story.append(Paragraph(f"Product: <font face='Courier'>{prod}</font>", body_style))
            story.append(
                Paragraph(
                    f"SMARTS: <font face='Courier'>{_truncate(step.smarts, 300)}</font>",
                    body_style,
                )
            )
            story.append(Spacer(1, 0.15 * inch))
            if step.image is not None:
                story.append(_pil_to_flowable(step.image, usable_w, usable_h * 0.65))

        if route_i < len(routes) - 1:
            story.append(PageBreak())

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title="ReaSyn pathway report",
    )
    doc.build(story)
    return path

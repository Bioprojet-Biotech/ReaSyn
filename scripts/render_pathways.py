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

import hashlib
import html
import pickle
import re
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

sys.path.append(".")

import click
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D

from reasyn.chem.matrix import ReactantReactionMatrix
from reasyn.chem.mol import Molecule
from reasyn.chem.stack import Stack

RXN_TOKEN = re.compile(r"^R(\d+)$")


@dataclass(frozen=True)
class ReactionStep:
    rxn_id: int
    smarts: str
    reactants: tuple[Molecule, ...]
    product: Molecule | None


def _load_rxn_matrix(matrix_path: Path) -> ReactantReactionMatrix:
    with matrix_path.open("rb") as handle:
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
    final_smiles: str,
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


def _prepare_mol(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    return mol


def _build_reaction(step: ReactionStep) -> rdChemReactions.ChemicalReaction:
    reactant_mols = [mol._rdmol for mol in step.reactants if mol._rdmol is not None]
    product_mol = step.product._rdmol if step.product is not None else None

    if reactant_mols and product_mol is not None:
        rxn = rdChemReactions.ChemicalReaction()
        for mol in reactant_mols:
            AllChem.Compute2DCoords(mol)
            rxn.AddReactantTemplate(mol)
        AllChem.Compute2DCoords(product_mol)
        rxn.AddProductTemplate(product_mol)
        rdChemReactions.Compute2DCoordsForReaction(rxn)
        return rxn

    template = AllChem.ReactionFromSmarts(step.smarts)
    rdChemReactions.Compute2DCoordsForReaction(template)
    return template


def _reaction_atom_counts(rxn: rdChemReactions.ChemicalReaction) -> list[int]:
    atom_counts: list[int] = []
    for i in range(rxn.GetNumReactantTemplates()):
        atom_counts.append(rxn.GetReactantTemplate(i).GetNumAtoms())
    for i in range(rxn.GetNumAgentTemplates()):
        atom_counts.append(rxn.GetAgentTemplate(i).GetNumAtoms())
    for i in range(rxn.GetNumProductTemplates()):
        atom_counts.append(rxn.GetProductTemplate(i).GetNumAtoms())
    return atom_counts


def _panel_height(
    atom_counts: list[int],
    panel_height: int,
    max_panel_height: int,
    n_extra_components: int = 0,
) -> int:
    if not atom_counts:
        return panel_height

    max_atoms = max(atom_counts)
    n_components = len(atom_counts) + n_extra_components
    # Grow height only when complexity warrants it, but always fill the chosen panel.
    suggested = panel_height + max(0, max_atoms - 28) * 3 + max(0, n_components - 2) * 18
    return min(max_panel_height, max(panel_height, suggested))


def _draw_reaction_step(
    step: ReactionStep,
    max_width: int,
    panel_height: int,
    max_panel_height: int,
) -> Image.Image:
    rxn = _build_reaction(step)
    atom_counts = _reaction_atom_counts(rxn)
    height = _panel_height(atom_counts, panel_height, max_panel_height)

    draw_opts = rdMolDraw2D.MolDrawOptions()
    draw_opts.padding = 0.02
    draw_opts.fixedBondLength = -1
    draw_opts.fixedScale = -1

    drawer = rdMolDraw2D.MolDraw2DCairo(max_width, height)
    drawer.SetDrawOptions(draw_opts)
    drawer.DrawReaction(rxn)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


def _wrap_text(text: str, max_chars: int) -> list[str]:
    if not text:
        return [""]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def _reaction_step_caption(step_idx: int, step: ReactionStep, wrap_chars: int) -> list[str]:
    n_r = len(step.reactants)
    reactant_label = f"{n_r} reactant" + ("s" if n_r != 1 else "")
    header = f"Step {step_idx}: R{step.rxn_id} ({reactant_label})"

    if ">>" in step.smarts:
        left, right = step.smarts.split(">>", 1)
        detail = f"{left.strip()} >> {right.strip()}"
    else:
        detail = step.smarts

    lines = [header, *_wrap_text(detail, wrap_chars)]
    return lines


def _annotate_image(image: Image.Image, title: str | list[str], width: int) -> Image.Image:
    lines = [title] if isinstance(title, str) else title
    font = ImageFont.load_default()
    line_height = 14
    padding = 8
    header_h = padding * 2 + line_height * len(lines)

    canvas = Image.new("RGB", (width, image.height + header_h), "white")
    canvas.paste(image, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill="black", font=font)
        y += line_height
    return canvas


def _stack_images(images: list[Image.Image], width: int) -> Image.Image:
    height = sum(image.height for image in images)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    return canvas


def _draw_overview(
    target: str,
    product: str,
    score: float,
    num_steps: int,
    width: int,
    panel_height: int,
    max_panel_height: int,
) -> Image.Image:
    mols = [_prepare_mol(target), _prepare_mol(product)]
    legends = [
        "Target",
        f"Product (score={score:.3f}, steps={num_steps})",
    ]
    valid = [(mol, legend) for mol, legend in zip(mols, legends) if mol is not None]
    if not valid:
        return Image.new("RGB", (width, 120), "white")

    atom_counts = [mol.GetNumAtoms() for mol, _ in valid]
    height = _panel_height(atom_counts, panel_height, max_panel_height)
    per_mol_width = width // len(valid)

    image = Draw.MolsToGridImage(
        [mol for mol, _ in valid],
        legends=[legend for _, legend in valid],
        molsPerRow=len(valid),
        subImgSize=(per_mol_width, height),
    )
    if image.width != width:
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    return _annotate_image(image, "Target vs proposed product", width)


def _safe_stem(smiles: str, index: int) -> str:
    digest = hashlib.md5(smiles.encode(), usedforsecurity=False).hexdigest()[:10]
    return f"pathway_{index:03d}_{digest}"


def render_pathway_row(
    row: pd.Series,
    rxn_matrix: ReactantReactionMatrix,
    output_dir: Path,
    width: int,
    panel_height: int,
    max_panel_height: int,
) -> tuple[Path, list[str]]:
    steps = pathway_to_steps(row["synthesis"], rxn_matrix, row["smiles"])
    step_summaries: list[str] = []
    wrap_chars = max(40, width // 8)
    images = [
        _draw_overview(
            target=row["target"],
            product=row["smiles"],
            score=float(row["score"]),
            num_steps=int(row["num_steps"]),
            width=width,
            panel_height=panel_height,
            max_panel_height=max_panel_height,
        )
    ]

    for step_idx, step in enumerate(steps, start=1):
        reaction_image = _draw_reaction_step(
            step,
            max_width=width,
            panel_height=panel_height,
            max_panel_height=max_panel_height,
        )
        caption = _reaction_step_caption(step_idx, step, wrap_chars)
        images.append(
            _annotate_image(
                reaction_image,
                caption,
                width=width,
            )
        )
        step_summaries.append(" ".join(caption))

    combined = _stack_images(images, width=width)
    output_path = output_dir / f"{_safe_stem(row['target'], int(row.name))}.png"
    combined.save(output_path)
    return output_path, step_summaries


def _write_html_index(rows: list[dict[str, str]], output_dir: Path) -> Path:
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


@click.command()
@click.argument("results_csv", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=Path("results/rendered"))
@click.option("--matrix-path", type=click.Path(exists=True, path_type=Path), default=Path("data/processed/comp_2048/matrix.pkl"))
@click.option("--top-k", type=int, default=1, show_default=True, help="Best K pathways per target.")
@click.option("--min-score", type=float, default=None, help="Optional minimum score filter.")
@click.option("--width", type=int, default=900, show_default=True)
@click.option("--panel-height", type=int, default=360, show_default=True, help="Height of each molecule/reaction panel.")
@click.option("--max-panel-height", type=int, default=520, show_default=True, help="Max auto-grown panel height for complex steps.")
def main(
    results_csv: Path,
    output: Path,
    matrix_path: Path,
    top_k: int,
    min_score: float | None,
    width: int,
    panel_height: int,
    max_panel_height: int,
) -> None:
    """Render ReaSyn CSV results as reaction diagrams (PNG + HTML index)."""
    df = pd.read_csv(results_csv)
    required = {"target", "smiles", "score", "synthesis", "num_steps"}
    missing = required - set(df.columns)
    if missing:
        raise click.ClickException(f"CSV missing columns: {sorted(missing)}")

    if min_score is not None:
        df = df[df["score"] >= min_score]

    df = df.sort_values(["target", "score"], ascending=[True, False])
    df = df.groupby("target", as_index=False).head(top_k)

    output.mkdir(parents=True, exist_ok=True)
    rxn_matrix = _load_rxn_matrix(matrix_path)

    html_rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        image_path, step_summaries = render_pathway_row(
            row=row,
            rxn_matrix=rxn_matrix,
            output_dir=output,
            width=width,
            panel_height=panel_height,
            max_panel_height=max_panel_height,
        )
        html_rows.append(
            {
                "target": str(row["target"]),
                "product": str(row["smiles"]),
                "score": f"{float(row['score']):.3f}",
                "num_steps": str(int(row["num_steps"])),
                "synthesis": str(row["synthesis"]),
                "image": image_path.name,
                "step_details": step_summaries,
            }
        )
        click.echo(f"Wrote {image_path}")

    index_path = _write_html_index(html_rows, output)
    click.echo(f"Wrote {index_path}")


if __name__ == "__main__":
    main()

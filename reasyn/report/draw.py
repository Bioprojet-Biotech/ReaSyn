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

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdChemReactions
from rdkit.Chem.Draw import rdMolDraw2D

from reasyn.report.types import DrawOptions, ReactionStep


def prepare_mol(smiles: str) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    return mol


def build_reaction(step: ReactionStep) -> rdChemReactions.ChemicalReaction:
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


def reaction_atom_counts(rxn: rdChemReactions.ChemicalReaction) -> list[int]:
    atom_counts: list[int] = []
    for i in range(rxn.GetNumReactantTemplates()):
        atom_counts.append(rxn.GetReactantTemplate(i).GetNumAtoms())
    for i in range(rxn.GetNumAgentTemplates()):
        atom_counts.append(rxn.GetAgentTemplate(i).GetNumAtoms())
    for i in range(rxn.GetNumProductTemplates()):
        atom_counts.append(rxn.GetProductTemplate(i).GetNumAtoms())
    return atom_counts


def panel_height(
    atom_counts: list[int],
    panel_height_px: int,
    max_panel_height: int,
    n_extra_components: int = 0,
) -> int:
    if not atom_counts:
        return panel_height_px

    max_atoms = max(atom_counts)
    n_components = len(atom_counts) + n_extra_components
    suggested = panel_height_px + max(0, max_atoms - 28) * 3 + max(0, n_components - 2) * 18
    return min(max_panel_height, max(panel_height_px, suggested))


def draw_reaction_step(step: ReactionStep, opts: DrawOptions) -> Image.Image:
    rxn = build_reaction(step)
    atom_counts = reaction_atom_counts(rxn)
    height = panel_height(atom_counts, opts.panel_height, opts.max_panel_height)

    draw_opts = rdMolDraw2D.MolDrawOptions()
    draw_opts.padding = 0.02
    draw_opts.fixedBondLength = -1
    draw_opts.fixedScale = -1

    drawer = rdMolDraw2D.MolDraw2DCairo(opts.width, height)
    drawer.SetDrawOptions(draw_opts)
    drawer.DrawReaction(rxn)
    drawer.FinishDrawing()
    return Image.open(BytesIO(drawer.GetDrawingText()))


def wrap_text(text: str, max_chars: int) -> list[str]:
    if not text:
        return [""]
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def reaction_step_caption(step_idx: int, step: ReactionStep, wrap_chars: int) -> list[str]:
    n_r = len(step.reactants)
    reactant_label = f"{n_r} reactant" + ("s" if n_r != 1 else "")
    header = f"Step {step_idx}: R{step.rxn_id} ({reactant_label})"

    if ">>" in step.smarts:
        left, right = step.smarts.split(">>", 1)
        detail = f"{left.strip()} >> {right.strip()}"
    else:
        detail = step.smarts

    return [header, *wrap_text(detail, wrap_chars)]


def annotate_image(image: Image.Image, title: str | list[str], width: int) -> Image.Image:
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


def stack_images(images: list[Image.Image], width: int) -> Image.Image:
    height = sum(image.height for image in images)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    return canvas


def draw_overview(
    target: str,
    product: str,
    score: float,
    num_steps: int,
    opts: DrawOptions,
) -> Image.Image:
    mols = [prepare_mol(target), prepare_mol(product)]
    legends = [
        "Target",
        f"Product (score={score:.3f}, steps={num_steps})",
    ]
    valid = [(mol, legend) for mol, legend in zip(mols, legends) if mol is not None]
    if not valid:
        return Image.new("RGB", (opts.width, 120), "white")

    atom_counts = [mol.GetNumAtoms() for mol, _ in valid]
    height = panel_height(atom_counts, opts.panel_height, opts.max_panel_height)
    per_mol_width = opts.width // len(valid)

    image = Draw.MolsToGridImage(
        [mol for mol, _ in valid],
        legends=[legend for _, legend in valid],
        molsPerRow=len(valid),
        subImgSize=(per_mol_width, height),
    )
    if image.width != opts.width:
        image = image.resize((opts.width, height), Image.Resampling.LANCZOS)
    return annotate_image(image, "Target vs proposed product", opts.width)


def placeholder_image(width: int, height: int, message: str) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((16, height // 2), message, fill="black", font=font)
    return canvas

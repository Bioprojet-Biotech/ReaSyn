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
sys.path.append('.')
import pathlib
from time import time

import click
import pandas as pd

from reasyn.chem.mol import Molecule, read_mol_file
from reasyn.sampler.parallel import run_sampling_one


def _input_mols_option(p):
    return list(read_mol_file(p))


def save_dataframe(df: pd.DataFrame, output: pathlib.Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix in (".feather", ".fth"):
        df.to_feather(output)
    elif suffix == ".csv":
        df.to_csv(output, index=False)
    elif suffix in (".xlsx", ".xls"):
        try:
            df.to_excel(output, index=False)
        except ImportError as e:
            raise click.ClickException(
                "Writing Excel requires openpyxl. Install with: uv add openpyxl"
            ) from e
    else:
        raise click.BadParameter(
            f"Unsupported output extension '{suffix}'. Use .feather, .fth, .csv, or .xlsx",
            param_hint="'--output' / '-o'",
        )
    click.echo(f"Saved results to {output}")


@click.command()
@click.option("--smiles", "-s", type=str, required=True)
@click.option("--output", "-o", type=click.Path(exists=False, path_type=pathlib.Path), default=None)
@click.option(
    "--model_path",
    "-m",
    type=str,
    default="checkpoints/nv-reasyn-ar-166m-v2.ckpt,checkpoints/nv-reasyn-eb-174m-v2.ckpt",
    show_default=True,
)
@click.option("--search_width", type=int, default=2)
@click.option("--exhaustiveness", type=int, default=4)
@click.option("--time_limit", type=int, default=1000)
@click.option("--add_bb_path", type=str, default=None)
@click.option("--num_cycles", type=int, default=2)
@click.option("--num_editflow_samples", type=int, default=10)
@click.option("--num_editflow_steps", type=int, default=100)
@click.option("--mols_to_filter", type=str, default=None)
@click.option("--filter_sim", type=float, default=0.8)
@click.option("--min_sim", type=float, default=0.0)
def main(
    smiles: str,
    output: pathlib.Path | None,
    model_path: str,
    search_width: int,
    exhaustiveness: int,
    time_limit: int,
    add_bb_path: str | None,
    num_cycles: int,
    num_editflow_samples: int,
    num_editflow_steps: int,
    mols_to_filter: str | None,
    filter_sim: float,
    min_sim: float,
):
    model_paths = [pathlib.Path(path) for path in model_path.split(",")]
    assert all(path.exists() for path in model_paths)

    t_start = time()
    df = run_sampling_one(
        input=Molecule(smiles),
        model_path=model_paths,
        search_width=search_width,
        exhaustiveness=exhaustiveness,
        time_limit=time_limit,
        add_bb_path=add_bb_path,
        num_cycles=num_cycles,
        num_editflow_samples=num_editflow_samples,
        num_editflow_steps=num_editflow_steps,
        mols_to_filter=_input_mols_option(mols_to_filter) if mols_to_filter is not None else None,
        filter_sim=filter_sim,
        min_sim=min_sim,
    )
    print(df)
    print(f"{time() - t_start:.2f} sec elapsed")

    if output is not None:
        save_dataframe(df, output)


if __name__ == "__main__":
    main()

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

import pandas as pd


def select_routes(
    df: pd.DataFrame,
    *,
    indices: list[int] | None = None,
    top_k: int | None = 1,
    min_score: float | None = None,
) -> pd.DataFrame:
    """Select pathway rows for reporting.

    If ``indices`` is set, ``min_score`` is applied first (when given), then the
    listed 0-based positions are taken from that filtered frame. Otherwise rows
    are filtered by ``min_score``, sorted by target then score descending, and
    ``top_k`` best pathways are kept per target (``top_k=None`` keeps all).
    """
    out = df.reset_index(drop=True)

    if min_score is not None:
        out = out[out["score"] >= min_score].reset_index(drop=True)

    if indices is not None:
        bad = [i for i in indices if i < 0 or i >= len(out)]
        if bad:
            raise IndexError(
                f"Invalid route indices {bad} for table of length {len(out)}"
            )
        return out.iloc[indices].reset_index(drop=True)

    out = out.sort_values(["target", "score"], ascending=[True, False])
    if top_k is None:
        return out.reset_index(drop=True)
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    return out.groupby("target", as_index=False).head(top_k).reset_index(drop=True)

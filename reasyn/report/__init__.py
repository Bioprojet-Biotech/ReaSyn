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

"""Pathway reporting: parse sampling tables into HTML/PNG or PDF documents."""

from reasyn.report.document import build_route, build_routes
from reasyn.report.generate import generate_pathways_report, parse_report_arg, write_report_from_arg
from reasyn.report.html import write_html_report
from reasyn.report.io import load_results_table
from reasyn.report.pdf import write_pdf
from reasyn.report.select import select_routes
from reasyn.report.steps import load_rxn_matrix, pathway_to_steps
from reasyn.report.types import DrawOptions, ReactionStep, RouteDocument, StepPage

__all__ = [
    "DrawOptions",
    "ReactionStep",
    "RouteDocument",
    "StepPage",
    "build_route",
    "build_routes",
    "generate_pathways_report",
    "load_results_table",
    "load_rxn_matrix",
    "parse_report_arg",
    "pathway_to_steps",
    "select_routes",
    "write_html_report",
    "write_pdf",
    "write_report_from_arg",
]

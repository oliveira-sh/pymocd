from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

_run_id = os.environ.get("BENCHMARK_RUN_ID") or datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)
SAVE_PATH = (
    str(Path(__file__).resolve().parent.parent / "results" / _run_id) + "/"
)

EXPORT_FORMATS = ("pdf", "png", "svg")

PALETTE = (
    "#2a78d6",
    "#eb6834",
    "#304e96",
    "#1baf7a",
    "#833a06",
    "#eda100",
    "#10632d",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
    "#009db8",
)


def algorithm_colors(names: Iterable[str]) -> dict[str, str]:
    ordered = list(dict.fromkeys(names))
    if len(ordered) > len(PALETTE):
        raise ValueError(
            f"Got {len(ordered)} algorithms but the palette has only "
            f"{len(PALETTE)} slots"
        )
    return {name: PALETTE[index] for index, name in enumerate(ordered)}


@dataclass(frozen=True)
class PlotTheme:
    figure_face: str
    axes_face: str
    text: str
    muted_text: str
    grid_major: str
    grid_minor: str
    spine: str
    legend_face: str
    legend_edge: str
    annotation_face: str
    annotation_edge: str
    accent: str
    edge_color: str
    q_cmap: str
    band_alpha: float
    network_palette: tuple[str, ...]
    heatmap_colors: tuple[str, str, str]


THEME = PlotTheme(
    figure_face="#ffffff",
    axes_face="#ffffff",
    text="#182230",
    muted_text="#5c6878",
    grid_major="#d8dfeb",
    grid_minor="#ebeff6",
    spine="#c5cfdd",
    legend_face="#ffffff",
    legend_edge="#d9e0ec",
    annotation_face="#ffffff",
    annotation_edge="#d9e0ec",
    accent="#0f6fff",
    edge_color="#8b98aa",
    q_cmap="cividis",
    band_alpha=0.14,
    network_palette=(
        "#2878b5",
        "#f28e2b",
        "#1f9d8a",
        "#d1495b",
        "#7b66d2",
        "#8f5b3f",
        "#4c956c",
        "#a23b72",
        "#d17a22",
        "#4b6cb7",
        "#6b8f71",
        "#b65f73",
    ),
    heatmap_colors=("#0f6fff", "#9eaabe", "#ecf0f6"),
)

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

_run_id = os.environ.get("BENCHMARK_RUN_ID") or datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)
SAVE_PATH = (
    str(
        Path(__file__).resolve().parent.parent.parent.parent
        / "tests"
        / "outputs"
        / _run_id
    )
    + "/"
)

EXPORT_FORMATS = ("pdf", "png", "svg")
DEFAULT_THEME_NAMES = ("light", "dark")
ALGORITHM_ORDER = ("PRISM", "HPMOCD", "SMPSO", "Leiden", "Louvain")


@dataclass(frozen=True)
class PlotTheme:
    name: str
    display_name: str
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
    algorithm_colors: Mapping[str, str]
    fallback_palette: tuple[str, ...]
    network_palette: tuple[str, ...]
    heatmap_colors: tuple[str, str, str]


LIGHT_THEME = PlotTheme(
    name="light",
    display_name="Light",
    figure_face="#f5f7fb",
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
    algorithm_colors={
        "PRISM": "#6f3cc2",
        "HPMOCD": "#1f9d8a",
        "SMPSO": "#d1495b",
        "Leiden": "#2878b5",
        "Louvain": "#f28e2b",
    },
    fallback_palette=(
        "#2878b5",
        "#f28e2b",
        "#1f9d8a",
        "#d1495b",
        "#7b66d2",
        "#8f5b3f",
        "#4c956c",
        "#a23b72",
    ),
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

DARK_THEME = PlotTheme(
    name="dark",
    display_name="Dark",
    figure_face="#0b1020",
    axes_face="#121a2a",
    text="#e7edf5",
    muted_text="#acb6c6",
    grid_major="#344157",
    grid_minor="#243044",
    spine="#43526b",
    legend_face="#10192a",
    legend_edge="#2b3a52",
    annotation_face="#172234",
    annotation_edge="#32425e",
    accent="#7cc3ff",
    edge_color="#7d8ea4",
    q_cmap="cividis",
    band_alpha=0.18,
    algorithm_colors={
        "PRISM": "#b08cff",
        "HPMOCD": "#46d1ba",
        "SMPSO": "#ff7a88",
        "Leiden": "#74b7ff",
        "Louvain": "#ffb463",
    },
    fallback_palette=(
        "#74b7ff",
        "#ffb463",
        "#46d1ba",
        "#ff7a88",
        "#a996ff",
        "#c8a57e",
        "#7ad18a",
        "#ffa5c2",
    ),
    network_palette=(
        "#74b7ff",
        "#ffb463",
        "#46d1ba",
        "#ff7a88",
        "#a996ff",
        "#c8a57e",
        "#7ad18a",
        "#ffa5c2",
        "#ffd166",
        "#62b6cb",
        "#c7f464",
        "#e9a03b",
    ),
    heatmap_colors=("#7cc3ff", "#4f6077", "#172233"),
)

THEMES = {theme.name: theme for theme in (LIGHT_THEME, DARK_THEME)}

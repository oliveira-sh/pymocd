from __future__ import annotations

import itertools
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, FormatStrFormatter

from .config import (
    ALGORITHM_ORDER,
    DEFAULT_THEME_NAMES,
    EXPORT_FORMATS,
    SAVE_PATH,
    THEMES,
    PlotTheme,
)

METRIC_METADATA = {
    "nmi": {"label": "NMI", "direction": "higher"},
    "ami": {"label": "AMI", "direction": "higher"},
    "modularity": {"label": "Modularity (Q)", "direction": "higher"},
    "time": {"label": "Runtime (s)", "direction": "lower"},
}
MARKER_OVERRIDES = {"HPMOCD": "^", "SMPSO": "D", "Leiden": "o", "Louvain": "s"}
MARKER_FALLBACK = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h")
SUMMARY_METRICS = ("nmi", "ami", "modularity", "time")


def resolve_themes(theme_names: Sequence[str] | None = None) -> list[PlotTheme]:
    names = theme_names or DEFAULT_THEME_NAMES
    return [THEMES[name] for name in names]


def theme_output_dir(save_path: str | Path, theme: PlotTheme) -> Path:
    return Path(save_path) / theme.name


def _theme_rc_params(theme: PlotTheme) -> dict[str, object]:
    return {
        "figure.facecolor": theme.figure_face,
        "savefig.facecolor": theme.figure_face,
        "savefig.edgecolor": theme.figure_face,
        "axes.facecolor": theme.axes_face,
        "axes.edgecolor": theme.spine,
        "axes.labelcolor": theme.text,
        "axes.titlecolor": theme.text,
        "text.color": theme.text,
        "xtick.color": theme.muted_text,
        "ytick.color": theme.muted_text,
        "grid.color": theme.grid_major,
        "grid.alpha": 1.0,
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "font.size": 10.5,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "figure.titlesize": 16,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.2,
        "lines.markersize": 6.0,
        "patch.linewidth": 0.8,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.minor.size": 3.0,
        "ytick.minor.size": 3.0,
        "savefig.dpi": 320,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }


@contextmanager
def theme_context(theme: PlotTheme):
    with plt.rc_context(_theme_rc_params(theme)):
        yield


def export_figure(
    fig: plt.Figure,
    save_path: str | Path,
    name: str,
    theme: PlotTheme,
) -> None:
    output_dir = theme_output_dir(save_path, theme)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in EXPORT_FORMATS:
        fig.savefig(
            output_dir / f"{name}.{fmt}",
            format=fmt,
            bbox_inches="tight",
            pad_inches=0.12,
            facecolor=theme.figure_face,
        )
    plt.close(fig)


def add_panel_tag(ax: plt.Axes, tag: str, theme: PlotTheme) -> None:
    ax.text(
        -0.06,
        1.14,
        tag,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=theme.muted_text,
        bbox={
            "boxstyle": "round,pad=0.22,rounding_size=0.18",
            "facecolor": theme.annotation_face,
            "edgecolor": theme.annotation_edge,
            "linewidth": 0.8,
        },
        clip_on=False,
    )


def _x_label(x_var: str) -> str:
    return "Mixing parameter (μ)" if x_var == "mu" else "Number of nodes"


def _format_nodes_tick(value: float, _position: int) -> str:
    if abs(value) >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{int(value)}"


def _sort_algorithms(algorithms: Iterable[str]) -> list[str]:
    preferred_order = {name: index for index, name in enumerate(ALGORITHM_ORDER)}
    return sorted(
        algorithms,
        key=lambda name: (preferred_order.get(name, len(preferred_order)), name),
    )


def build_algorithm_styles(
    algorithms: Iterable[str], theme: PlotTheme
) -> Dict[str, tuple[str, str]]:
    ordered_algorithms = _sort_algorithms(algorithms)
    markers = itertools.cycle(MARKER_FALLBACK)
    palette = itertools.cycle(theme.fallback_palette)
    styles: Dict[str, tuple[str, str]] = {}
    for algorithm in ordered_algorithms:
        styles[algorithm] = (
            theme.algorithm_colors.get(algorithm, next(palette)),
            MARKER_OVERRIDES.get(algorithm, next(markers)),
        )
    return styles


def _prepare_results_frame(results: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
    df = pd.DataFrame(results).copy() if isinstance(results, dict) else results.copy()
    rename_map = {
        "modularity_mean": "modularity",
        "nmi_mean": "nmi",
        "ami_mean": "ami",
        "time_mean": "time",
    }
    return df.rename(columns=rename_map)


def infer_x_var(df: pd.DataFrame) -> str:
    if "mu" in df.columns:
        return "mu"
    if "nodes" in df.columns:
        return "nodes"
    raise ValueError("Neither 'mu' nor 'nodes' found in benchmark results")


def _legend_frame(legend: plt.Legend, theme: PlotTheme) -> None:
    frame = legend.get_frame()
    frame.set_facecolor(theme.legend_face)
    frame.set_edgecolor(theme.legend_edge)
    frame.set_linewidth(0.8)
    frame.set_alpha(1.0)


def style_axis(
    ax: plt.Axes,
    *,
    theme: PlotTheme,
    x_label: str,
    y_label: str,
    x_var: str | None = None,
    x_values: Sequence[float] | None = None,
    log_scale: str | None = None,
) -> None:
    ax.set_xlabel(x_label, labelpad=10)
    ax.set_ylabel(y_label, labelpad=10)
    if log_scale == "y":
        ax.set_yscale("log")
    elif log_scale == "x":
        ax.set_xscale("log")
    elif log_scale == "both":
        ax.set_xscale("log")
        ax.set_yscale("log")

    if x_var == "mu":
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    elif x_var == "nodes":
        ax.xaxis.set_major_formatter(FuncFormatter(_format_nodes_tick))

    if x_values is not None and len(x_values) <= 10:
        ax.set_xticks(sorted({float(value) for value in x_values}))

    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", linewidth=0.9, color=theme.grid_major)
    ax.grid(axis="x", which="major", linewidth=0.7, color=theme.grid_minor)

    if log_scale == "y":
        ax.grid(axis="y", which="minor", linewidth=0.5, color=theme.grid_minor)
    else:
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(axis="y", which="minor", linewidth=0.5, color=theme.grid_minor)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(theme.spine)
    ax.spines["bottom"].set_color(theme.spine)
    ax.tick_params(axis="both", which="both", pad=4)
    ax.margins(x=0.03)


def _plot_metric_series(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    x_var: str,
    styles: Mapping[str, tuple[str, str]],
    theme: PlotTheme,
    *,
    with_labels: bool,
) -> None:
    for algorithm in _sort_algorithms(df["algorithm"].unique()):
        algorithm_data = df[df["algorithm"] == algorithm].sort_values(by=x_var)
        x_values = algorithm_data[x_var].to_numpy(dtype=float)
        y_values = algorithm_data[metric].to_numpy(dtype=float)
        color, marker = styles[algorithm]

        ax.plot(
            x_values,
            y_values,
            color=color,
            marker=marker,
            label=algorithm if with_labels else None,
            linewidth=2.3,
            markersize=6.2,
            markerfacecolor=color,
            markeredgecolor=theme.axes_face,
            markeredgewidth=1.2,
            solid_capstyle="round",
            zorder=3,
        )

        std_key = f"{metric}_std"
        if std_key in algorithm_data.columns and not algorithm_data[std_key].isna().all():
            y_std = algorithm_data[std_key].fillna(0).to_numpy(dtype=float)
            lower = y_values - y_std
            upper = y_values + y_std
            if metric == "time":
                lower = np.clip(lower, np.finfo(float).eps, None)
            ax.fill_between(
                x_values,
                lower,
                upper,
                color=color,
                alpha=theme.band_alpha,
                linewidth=0,
                zorder=2,
            )


def plot_single_metric(
    df: pd.DataFrame,
    metric: str,
    x_var: str,
    save_path: str = SAVE_PATH,
    theme_names: Sequence[str] | None = None,
) -> None:
    algorithms = _sort_algorithms(df["algorithm"].unique())
    x_values = sorted(df[x_var].astype(float).unique())

    for theme in resolve_themes(theme_names):
        styles = build_algorithm_styles(algorithms, theme)
        with theme_context(theme):
            fig, ax = plt.subplots(figsize=(8.6, 5.1))
            _plot_metric_series(ax, df, metric, x_var, styles, theme, with_labels=True)
            style_axis(
                ax,
                theme=theme,
                x_label=_x_label(x_var),
                y_label=METRIC_METADATA[metric]["label"],
                x_var=x_var,
                x_values=x_values,
                log_scale="y" if metric == "time" else None,
            )
            legend = ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=min(len(algorithms), 4),
                columnspacing=1.1,
                handlelength=2.1,
                borderpad=0.6,
            )
            _legend_frame(legend, theme)
            fig.tight_layout()
            export_figure(fig, save_path, f"{metric}_plot", theme)


def plot_comparison_matrix(
    df: pd.DataFrame,
    x_var: str,
    save_path: str = SAVE_PATH,
    theme_names: Sequence[str] | None = None,
) -> None:
    algorithms = _sort_algorithms(df["algorithm"].unique())
    x_values = sorted(df[x_var].astype(float).unique())
    metrics = list(SUMMARY_METRICS)

    for theme in resolve_themes(theme_names):
        styles = build_algorithm_styles(algorithms, theme)
        with theme_context(theme):
            fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), sharex="col")
            axes = axes.flatten()

            for index, metric in enumerate(metrics):
                ax = axes[index]
                _plot_metric_series(
                    ax,
                    df,
                    metric,
                    x_var,
                    styles,
                    theme,
                    with_labels=index == 0,
                )
                add_panel_tag(ax, f"{chr(97 + index)})", theme)
                style_axis(
                    ax,
                    theme=theme,
                    x_label=_x_label(x_var),
                    y_label=METRIC_METADATA[metric]["label"],
                    x_var=x_var,
                    x_values=x_values,
                    log_scale="y" if metric == "time" else None,
                )

            handles, labels = axes[0].get_legend_handles_labels()
            legend = fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.55, 0.02),
                ncol=min(len(algorithms), 4),
                columnspacing=1.5,
                handlelength=2.2,
            )
            _legend_frame(legend, theme)
            fig.subplots_adjust(top=0.94, bottom=0.12, hspace=0.34, wspace=0.24)
            export_figure(fig, save_path, "comparison_matrix", theme)


def _performance_rank_table(df: pd.DataFrame, x_var: str) -> pd.DataFrame:
    rank_rows = []
    for _, subset in df.groupby(x_var):
        indexed = subset.set_index("algorithm")
        for metric in SUMMARY_METRICS:
            ascending = METRIC_METADATA[metric]["direction"] == "lower"
            metric_ranks = indexed[metric].rank(ascending=ascending, method="average")
            for algorithm, rank_value in metric_ranks.items():
                rank_rows.append(
                    {"algorithm": algorithm, "metric": metric, "rank": rank_value}
                )

    rank_frame = pd.DataFrame(rank_rows)
    summary = rank_frame.pivot_table(
        index="algorithm",
        columns="metric",
        values="rank",
        aggfunc="mean",
    )
    summary = summary.reindex(_sort_algorithms(summary.index))
    summary["overall"] = summary[list(SUMMARY_METRICS)].mean(axis=1)
    return summary


def _rank_heatmap_cmap(theme: PlotTheme) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f"benchmark_rank_{theme.name}",
        list(theme.heatmap_colors),
    )


def _relative_luminance(color: tuple[float, float, float, float]) -> float:
    red, green, blue = color[:3]
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def plot_performance_scorecard(
    df: pd.DataFrame,
    x_var: str,
    save_path: str = SAVE_PATH,
    theme_names: Sequence[str] | None = None,
) -> None:
    rank_table = _performance_rank_table(df, x_var)
    algorithms = list(rank_table.index)
    columns = ["nmi", "ami", "modularity", "time", "overall"]
    labels = ["NMI ↑", "AMI ↑", "Q ↑", "Time ↓", "Overall ↓"]
    values = rank_table[columns].to_numpy(dtype=float)
    vmin = 1.0
    vmax = float(len(algorithms))

    for theme in resolve_themes(theme_names):
        cmap = _rank_heatmap_cmap(theme)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        with theme_context(theme):
            fig, ax = plt.subplots(figsize=(8.3, 4.6))
            image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
            ax.set_xticks(np.arange(len(columns)))
            ax.set_xticklabels(labels)
            ax.set_yticks(np.arange(len(algorithms)))
            ax.set_yticklabels(algorithms)
            ax.tick_params(axis="x", rotation=0)
            ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(algorithms), 1), minor=True)
            ax.grid(which="minor", color=theme.figure_face, linewidth=1.4)
            ax.tick_params(which="minor", bottom=False, left=False)

            for spine in ax.spines.values():
                spine.set_color(theme.spine)
                spine.set_linewidth(0.8)

            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    cell_value = values[row, col]
                    color = cmap(norm(cell_value))
                    text_color = (
                        "#f8fbff"
                        if _relative_luminance(color) < 0.46
                        else theme.text
                    )
                    weight = (
                        "semibold"
                        if np.isclose(cell_value, np.nanmin(values[:, col]))
                        else "normal"
                    )
                    ax.text(
                        col,
                        row,
                        f"{cell_value:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontweight=weight,
                        fontsize=9.5,
                    )

            colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
            colorbar.outline.set_edgecolor(theme.spine)
            colorbar.outline.set_linewidth(0.8)
            colorbar.ax.tick_params(colors=theme.muted_text, labelsize=8.8)
            colorbar.set_label("Average rank", color=theme.text, fontsize=9.5)
            fig.tight_layout()
            export_figure(fig, save_path, "performance_scorecard", theme)


def plot_results(
    results: Union[Dict, pd.DataFrame],
    save_path: str = SAVE_PATH,
    theme_names: Sequence[str] | None = None,
) -> None:
    df = _prepare_results_frame(results)
    x_var = infer_x_var(df)

    for metric in SUMMARY_METRICS:
        if metric in df.columns:
            plot_single_metric(df, metric, x_var, save_path, theme_names)

    plot_comparison_matrix(df, x_var, save_path, theme_names)
    if all(metric in df.columns for metric in ("nmi", "ami", "modularity", "time")):
        plot_performance_scorecard(df, x_var, save_path, theme_names)

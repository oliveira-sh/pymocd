import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union

from .config import SAVE_PATH

_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
_MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "x"]


def _get_style(algorithms):
    colors = itertools.cycle(_PALETTE)
    markers = itertools.cycle(_MARKERS)
    return {alg: (next(colors), next(markers)) for alg in algorithms}


def _style_axis(ax, x_label, y_label, title=None, log_scale=None):
    ax.set_xlabel(x_label, fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold", pad=20)
    if log_scale == "y":
        ax.set_yscale("log")
    elif log_scale == "x":
        ax.set_xscale("log")
    elif log_scale == "both":
        ax.set_yscale("log")
        ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.7, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", length=4, width=1.0)


_Y_LABELS = {
    "nmi": "NMI",
    "ami": "AMI",
    "modularity": "Modularity",
    "time": "Execution time (s)",
}


def _x_label(x_var):
    return "Mixing parameter (μ)" if x_var == "mu" else "Number of nodes (n)"


def _save(fig, output_path, name):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for fmt in ("pdf", "png"):
        fig.savefig(
            output_path / f"{name}.{fmt}",
            format=fmt,
            bbox_inches="tight",
            dpi=300,
            facecolor="white",
        )
    plt.close(fig)


def plot_single_metric(
    df: pd.DataFrame, metric: str, x_var: str, save_path: str = SAVE_PATH
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    algorithms = df["algorithm"].unique()
    styles = _get_style(algorithms)

    for alg in algorithms:
        alg_data = df[df["algorithm"] == alg].sort_values(by=x_var)
        x_values = alg_data[x_var].values
        y_values = alg_data[metric].values
        color, marker = styles[alg]

        ax.plot(
            x_values,
            y_values,
            color=color,
            marker=marker,
            label=alg,
            linewidth=2.5,
            markersize=8,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=1.5,
        )

        std_key = f"{metric}_std"
        if std_key in alg_data.columns and not alg_data[std_key].isna().all():
            y_std = alg_data[std_key].values
            ax.fill_between(
                x_values,
                y_values - y_std,
                y_values + y_std,
                color=color,
                alpha=0.2,
                interpolate=True,
            )

    _style_axis(ax, _x_label(x_var), _Y_LABELS.get(metric, metric.upper()))

    legend = ax.legend(
        loc="best",
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=2 if len(algorithms) > 6 else 1,
        handlelength=2.5,
        handletextpad=0.8,
        columnspacing=1.0,
        borderaxespad=0.5,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor("gray")
    legend.get_frame().set_linewidth(1.0)

    plt.tight_layout(pad=1.0)
    _save(fig, save_path, f"{metric}_plot")


def plot_comparison_matrix(
    df: pd.DataFrame, x_var: str, save_path: str = SAVE_PATH
) -> None:
    metrics = ["nmi", "ami", "modularity", "time"]
    y_labels = {
        "nmi": "NMI",
        "ami": "AMI",
        "modularity": "Modularity (Q)",
        "time": "Time (s)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    algorithms = df["algorithm"].unique()
    styles = _get_style(algorithms)

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for alg in algorithms:
            alg_data = df[df["algorithm"] == alg].sort_values(by=x_var)
            x_values = alg_data[x_var].values
            y_values = alg_data[metric].values
            color, marker = styles[alg]

            ax.plot(
                x_values,
                y_values,
                color=color,
                marker=marker,
                label=alg if idx == 0 else "",
                linewidth=2.0,
                markersize=6,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=1.0,
            )

            std_key = f"{metric}_std"
            if std_key in alg_data.columns and not alg_data[std_key].isna().all():
                y_std = alg_data[std_key].values
                ax.fill_between(
                    x_values,
                    y_values - y_std,
                    y_values + y_std,
                    color=color,
                    alpha=0.15,
                )

        log_scale = "y" if metric == "time" else None
        _style_axis(ax, _x_label(x_var), y_labels[metric], log_scale=log_scale)

        ax.text(
            0.02,
            0.98,
            f"({chr(97 + idx)})",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(algorithms), 5),
        frameon=True,
        fancybox=True,
        shadow=True,
        handlelength=2.0,
        handletextpad=0.8,
        columnspacing=1.5,
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    _save(fig, save_path, "comparison_matrix")


def plot_performance_radar(df: pd.DataFrame, save_path: str = SAVE_PATH) -> None:
    metrics = ["nmi", "ami", "modularity"]
    algorithms = df["algorithm"].unique()
    styles = _get_style(algorithms)

    normalized_data = {}
    for metric in metrics:
        max_val = df[metric].max()
        min_val = df[metric].min()
        for alg in algorithms:
            mean_val = df[df["algorithm"] == alg][metric].mean()
            normalized_val = (
                (mean_val - min_val) / (max_val - min_val)
                if max_val != min_val
                else 0.0
            )
            normalized_data.setdefault(alg, {})[metric] = normalized_val

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    for alg in algorithms:
        values = [normalized_data[alg][m] for m in metrics] + [
            normalized_data[alg][metrics[0]]
        ]
        color, _ = styles[alg]
        ax.plot(angles, values, "o-", linewidth=2, label=alg, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["NMI", "AMI", "Modularity"], fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
    ax.grid(True)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    _save(fig, save_path, "performance_radar")


def plot_results(
    results: Union[Dict, pd.DataFrame], save_path: str = SAVE_PATH
) -> None:
    if isinstance(results, dict):
        df = pd.DataFrame(results)
    else:
        df = results

    if "mu" in df.columns:
        x_var = "mu"
    elif "nodes" in df.columns:
        x_var = "nodes"
    else:
        raise ValueError("Neither 'mu' nor 'nodes' found in results")

    for metric in ("nmi", "ami", "modularity", "time"):
        if metric in df.columns:
            plot_single_metric(df, metric, x_var, save_path)

    plot_comparison_matrix(df, x_var, save_path)

    if all(m in df.columns for m in ("nmi", "ami", "modularity")):
        plot_performance_radar(df, save_path)

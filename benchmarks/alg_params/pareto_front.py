from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pymocd

from utils import SAVE_PATH, THEME, evaluate_communities, generate_lfr_benchmark
from utils.plotting import (
    add_panel_tag,
    export_figure,
    style_axis,
    theme_context,
)


def _partition_objectives(
    graph: nx.Graph, partition: dict[int, int]
) -> tuple[float, float]:
    num_edges = graph.number_of_edges()
    members: dict[int, set] = {}
    for node, community in partition.items():
        members.setdefault(community, set()).add(node)

    intra = 1.0
    inter = 0.0
    for nodes in members.values():
        intra -= graph.subgraph(nodes).number_of_edges() / num_edges
        degree_sum = sum(degree for _, degree in graph.degree(nodes))
        inter += (degree_sum / (2 * num_edges)) ** 2
    return intra, inter


def _compute_pareto_records() -> tuple[list[dict[str, object]], dict[str, object]]:
    graph, ground_truth = generate_lfr_benchmark()
    pareto_front = pymocd.scale_fronts(graph)
    records: list[dict[str, object]] = []

    for communities in pareto_front:
        intra, inter = _partition_objectives(graph, communities)
        metrics = evaluate_communities(graph, communities, ground_truth, convert=False)
        records.append(
            {
                "communities": communities,
                "intra": float(intra),
                "inter": float(inter),
                "q": float(1 - intra - inter),
                "num_communities": len(set(communities.values())),
                "modularity": float(metrics["modularity"]),
                "nmi": float(metrics["nmi"]),
                "ami": float(metrics["ami"]),
            }
        )

    best = max(records, key=lambda record: record["q"])
    return records, best


def _record_values(records: list[dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([record[key] for record in records], dtype=float)


def _plot_pareto_front(ax: plt.Axes, records, best) -> None:
    intra_values = _record_values(records, "intra")
    inter_values = _record_values(records, "inter")
    q_values = _record_values(records, "q")

    scatter = ax.scatter(
        intra_values,
        inter_values,
        c=q_values,
        cmap=THEME.q_cmap,
        s=72,
        alpha=0.95,
        edgecolors=THEME.figure_face,
        linewidths=0.9,
        zorder=3,
    )
    ax.scatter(
        best["intra"],
        best["inter"],
        s=240,
        marker="*",
        color=THEME.accent,
        edgecolors=THEME.axes_face,
        linewidths=1.2,
        zorder=5,
    )
    add_panel_tag(ax, "a)")
    style_axis(
        ax,
        x_label="Intra objective",
        y_label="Inter objective",
        x_values=intra_values,
    )
    colorbar = ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025)
    colorbar.outline.set_edgecolor(THEME.spine)
    colorbar.outline.set_linewidth(0.8)
    colorbar.ax.tick_params(colors=THEME.muted_text, labelsize=8.8)
    colorbar.set_label("Q score", color=THEME.text, fontsize=9.5)


def _plot_q_relationship(
    ax: plt.Axes,
    records: list[dict[str, object]],
    best: dict[str, object],
    x_key: str,
    x_label: str,
    tag: str,
) -> None:
    x_values = _record_values(records, x_key)
    q_values = _record_values(records, "q")
    ax.scatter(
        x_values,
        q_values,
        c=q_values,
        cmap=THEME.q_cmap,
        s=54,
        alpha=0.9,
        edgecolors=THEME.figure_face,
        linewidths=0.7,
        zorder=3,
    )
    ax.scatter(
        best[x_key],
        best["q"],
        s=98,
        marker="o",
        color=THEME.accent,
        edgecolors=THEME.axes_face,
        linewidths=1.1,
        zorder=4,
    )
    ax.axhline(best["q"], color=THEME.muted_text, linewidth=0.9, linestyle="--")
    add_panel_tag(ax, tag)
    style_axis(
        ax,
        x_label=x_label,
        y_label="Q score",
        x_values=x_values,
    )


def _build_dashboard(records, best) -> plt.Figure:
    fig = plt.figure(figsize=(13.8, 7.8))
    axes = fig.subplot_mosaic(
        [
            ["front", "communities", "modularity"],
            ["front", "nmi", "ami"],
        ]
    )

    _plot_pareto_front(axes["front"], records, best)
    _plot_q_relationship(
        axes["communities"],
        records,
        best,
        "num_communities",
        "Number of communities",
        "b)",
    )
    _plot_q_relationship(
        axes["modularity"],
        records,
        best,
        "modularity",
        "Modularity",
        "c)",
    )
    _plot_q_relationship(
        axes["nmi"],
        records,
        best,
        "nmi",
        "NMI",
        "d)",
    )
    _plot_q_relationship(
        axes["ami"],
        records,
        best,
        "ami",
        "AMI",
        "e)",
    )
    fig.subplots_adjust(top=0.96, wspace=0.28, hspace=0.32)
    return fig


def _build_front_only(records, best) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    _plot_pareto_front(ax, records, best)
    fig.tight_layout()
    return fig


def main() -> None:
    records, best = _compute_pareto_records()
    with theme_context():
        dashboard = _build_dashboard(records, best)
        export_figure(dashboard, SAVE_PATH, "pareto_frontier_analysis")

        front_only = _build_front_only(records, best)
        export_figure(front_only, SAVE_PATH, "pareto_front_plot")


if __name__ == "__main__":
    main()

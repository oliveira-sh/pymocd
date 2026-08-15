from __future__ import annotations

import os
import sys
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pymocd

from utils import SAVE_PATH, THEME, generate_lfr_benchmark
from utils.plotting import (
    add_panel_tag,
    export_figure,
    theme_context,
)

pymocd.max_cores(2)

GENERATIONS = [10, 30, 50, 80, 100, 110]


def _remap_partition(partition: dict[int, int]) -> tuple[dict[int, int], int]:
    grouped_nodes: dict[int, list[int]] = {}
    for node, community in partition.items():
        grouped_nodes.setdefault(int(community), []).append(int(node))

    ordered_groups = sorted(
        grouped_nodes.values(),
        key=lambda nodes: (-len(nodes), min(nodes)),
    )

    remapped: dict[int, int] = {}
    for new_label, nodes in enumerate(ordered_groups):
        for node in nodes:
            remapped[node] = new_label

    return remapped, len(ordered_groups)


def _community_cmap(num_communities: int) -> ListedColormap:
    colors = [
        THEME.network_palette[index % len(THEME.network_palette)]
        for index in range(max(num_communities, 1))
    ]
    return ListedColormap(colors)


def _style_network_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(THEME.axes_face)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_partition(
    ax: plt.Axes,
    graph: nx.Graph,
    pos: dict[int, tuple[float, float]],
    partition: dict[int, int],
    *,
    tag: str | None = None,
) -> None:
    normalized_partition, num_communities = _remap_partition(partition)
    node_values = np.asarray([normalized_partition[node] for node in graph.nodes()])
    cmap = _community_cmap(num_communities)

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color=THEME.edge_color,
        alpha=0.16,
        width=0.55,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_values,
        cmap=cmap,
        node_size=42,
        linewidths=0.45,
        edgecolors=THEME.axes_face,
    )

    if tag is not None:
        add_panel_tag(ax, tag)
    _style_network_axis(ax)


def _build_overview(
    graph: nx.Graph,
    pos: dict[int, tuple[float, float]],
    partitions: dict[int, dict[int, int]],
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.8))
    axes = axes.flatten()

    for index, generation in enumerate(GENERATIONS):
        _draw_partition(
            axes[index],
            graph,
            pos,
            partitions[generation],
            tag=f"{chr(97 + index)}) g={generation}",
        )

    fig.subplots_adjust(top=0.97, wspace=0.08, hspace=0.18)
    return fig


def _build_single_generation(
    graph: nx.Graph,
    pos: dict[int, tuple[float, float]],
    partition: dict[int, int],
    generation: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    _draw_partition(ax, graph, pos, partition)
    fig.tight_layout()
    return fig


def _compute_partitions(graph: nx.Graph) -> dict[int, dict[int, int]]:
    partitions: dict[int, dict[int, int]] = {}
    for generation in GENERATIONS:
        partitions[generation] = pymocd.gmocs(graph, num_gens=generation)
    return partitions


def main() -> None:
    graph, _ = generate_lfr_benchmark(n=250, mu=0.1)
    pos = nx.spring_layout(graph, seed=42)
    partitions = _compute_partitions(graph)

    with theme_context():
        overview = _build_overview(graph, pos, partitions)
        export_figure(overview, SAVE_PATH, "community_evolution_overview")

        for generation in GENERATIONS:
            figure = _build_single_generation(
                graph,
                pos,
                partitions[generation],
                generation,
            )
            export_figure(figure, SAVE_PATH, f"communities_gen_{generation}")


if __name__ == "__main__":
    main()

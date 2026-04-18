from __future__ import annotations

import os
import sys
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pymocd

from utils import SAVE_PATH, generate_lfr_benchmark  # noqa: E402
from utils.plotting import (  # noqa: E402
    add_panel_tag,
    export_figure,
    resolve_themes,
    theme_context,
)

pymocd.set_thread_count(2)

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


def _community_cmap(theme, num_communities: int) -> ListedColormap:
    colors = [
        theme.network_palette[index % len(theme.network_palette)]
        for index in range(max(num_communities, 1))
    ]
    return ListedColormap(colors)


def _style_network_axis(ax: plt.Axes, theme) -> None:
    ax.set_facecolor(theme.axes_face)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_partition(
    ax: plt.Axes,
    graph: nx.Graph,
    pos: dict[int, tuple[float, float]],
    partition: dict[int, int],
    theme,
    *,
    tag: str | None = None,
) -> None:
    normalized_partition, num_communities = _remap_partition(partition)
    node_values = np.asarray([normalized_partition[node] for node in graph.nodes()])
    cmap = _community_cmap(theme, num_communities)

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color=theme.edge_color,
        alpha=0.16 if theme.name == "light" else 0.12,
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
        edgecolors=theme.axes_face,
    )

    if tag is not None:
        add_panel_tag(ax, tag, theme)
    _style_network_axis(ax, theme)


def _build_overview(
    graph: nx.Graph,
    pos: dict[int, tuple[float, float]],
    partitions: dict[int, dict[int, int]],
    theme,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.8))
    axes = axes.flatten()

    for index, generation in enumerate(GENERATIONS):
        _draw_partition(
            axes[index],
            graph,
            pos,
            partitions[generation],
            theme,
            tag=f"{chr(97 + index)}) g={generation}",
        )

    fig.subplots_adjust(top=0.97, wspace=0.08, hspace=0.18)
    return fig


def _build_single_generation(
    graph: nx.Graph,
    pos: dict[int, tuple[float, float]],
    partition: dict[int, int],
    generation: int,
    theme,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    _draw_partition(ax, graph, pos, partition, theme)
    fig.tight_layout()
    return fig


def _compute_partitions(graph: nx.Graph) -> dict[int, dict[int, int]]:
    partitions: dict[int, dict[int, int]] = {}
    for generation in GENERATIONS:
        model = pymocd.HpMocd(graph, num_gens=generation)
        partitions[generation] = model.run()
    return partitions


def main() -> None:
    graph, _ = generate_lfr_benchmark(n=250, mu=0.1)
    pos = nx.spring_layout(graph, seed=42)
    partitions = _compute_partitions(graph)

    for theme in resolve_themes():
        with theme_context(theme):
            overview = _build_overview(graph, pos, partitions, theme)
            export_figure(overview, SAVE_PATH, "community_evolution_overview", theme)

            for generation in GENERATIONS:
                figure = _build_single_generation(
                    graph,
                    pos,
                    partitions[generation],
                    generation,
                    theme,
                )
                export_figure(figure, SAVE_PATH, f"communities_gen_{generation}", theme)


if __name__ == "__main__":
    main()

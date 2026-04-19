"""Real-world network benchmark.

Runs the configured algorithm registry (PRISM, Leiden, Louvain) against a
small zoo of real networks bundled with NetworkX / igraph. Graphs with a
known reference partition report NMI and AMI against that ground truth;
graphs without it report modularity + runtime only.

Output lands under ``<SAVE_PATH>/real_nets/``:
- ``real_nets.csv`` with per-(network, algorithm) means/stds
- grouped-bar plots per metric (modularity, nmi, ami, time) per theme
"""

from __future__ import annotations

import os
import sys
import time
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from utils import SAVE_PATH, evaluate_communities
from utils.lfr import convert_communities_to_partition
from utils.plotting import (
    METRIC_METADATA,
    _legend_frame,
    _sort_algorithms,
    build_algorithm_styles,
    export_figure,
    resolve_themes,
    theme_context,
)

# Populate the algorithm registry as a side effect of this import.
from lfr_experiment import ALGORITHM_REGISTRY  # noqa: F401
from lfr_experiment.registry import ALGORITHM_REGISTRY  # noqa: F811

N_RUNS = 3


def _relabel_int(G: nx.Graph, gt: dict | None = None) -> tuple[nx.Graph, dict | None]:
    """PRISM requires integer node IDs; normalize any string-labeled graph."""
    if all(isinstance(n, int) for n in G.nodes()):
        return G, gt
    mapping = {old: idx for idx, old in enumerate(sorted(G.nodes(), key=str))}
    H = nx.relabel_nodes(G, mapping, copy=True)
    if gt is not None:
        gt = {mapping[k]: v for k, v in gt.items()}
    return H, gt


def _karate() -> tuple[nx.Graph, dict[int, int]]:
    G = nx.karate_club_graph()
    gt = {n: 0 if G.nodes[n]["club"] == "Mr. Hi" else 1 for n in G.nodes()}
    return _relabel_int(G, gt)


def _les_miserables() -> tuple[nx.Graph, None]:
    G, _ = _relabel_int(nx.Graph(nx.les_miserables_graph()))
    return G, None


def _florentine() -> tuple[nx.Graph, None]:
    G = nx.florentine_families_graph()
    # Isolate "Pucci" has no edges → drop to keep algorithms happy.
    G = G.subgraph([n for n, d in G.degree() if d > 0]).copy()
    G, _ = _relabel_int(G)
    return G, None


NETWORKS: list[tuple[str, Callable[[], tuple[nx.Graph, dict | None]]]] = [
    ("Karate", _karate),
    ("Les Misérables", _les_miserables),
    ("Florentine", _florentine),
]


def _evaluate(
    G: nx.Graph,
    communities,
    gt: dict | None,
    convert: bool,
) -> dict[str, float | None]:
    if gt is not None:
        return evaluate_communities(G, communities, gt, convert=convert)

    partition = (
        convert_communities_to_partition(communities) if convert else communities
    )
    if not partition:
        return {"modularity": 0.0, "nmi": None, "ami": None}

    comms_dict: dict[int, set] = {}
    for node, comm in partition.items():
        comms_dict.setdefault(comm, set()).add(node)
    mod = nx.algorithms.community.modularity(G, list(comms_dict.values()))
    return {"modularity": float(mod), "nmi": None, "ami": None}


def run_all(n_runs: int = N_RUNS) -> pd.DataFrame:
    records: list[dict] = []
    for net_name, loader in NETWORKS:
        G, gt = loader()
        print(
            f"[{net_name}] n={G.number_of_nodes()} m={G.number_of_edges()} "
            f"gt={'yes' if gt is not None else 'no'}"
        )
        for alg_name, info in ALGORITHM_REGISTRY.items():
            alg_func = info["function"]
            needs_conv = info["needs_conversion"]
            mods, nmis, amis, durations = [], [], [], []
            for run_id in range(n_runs):
                start = time.perf_counter()
                communities = alg_func(G, seed=run_id)
                durations.append(time.perf_counter() - start)
                result = _evaluate(G, communities, gt, convert=needs_conv)
                mods.append(result["modularity"])
                if result["nmi"] is not None:
                    nmis.append(result["nmi"])
                    amis.append(result["ami"])

            row = {
                "network": net_name,
                "algorithm": alg_name,
                "n_nodes": int(G.number_of_nodes()),
                "n_edges": int(G.number_of_edges()),
                "modularity": float(np.mean(mods)),
                "modularity_std": float(np.std(mods)),
                "time": float(np.mean(durations)),
                "time_std": float(np.std(durations)),
                "nmi": float(np.mean(nmis)) if nmis else np.nan,
                "nmi_std": float(np.std(nmis)) if nmis else np.nan,
                "ami": float(np.mean(amis)) if amis else np.nan,
                "ami_std": float(np.std(amis)) if amis else np.nan,
            }
            records.append(row)
    return pd.DataFrame(records)


def _bar_plot(
    df: pd.DataFrame,
    metric: str,
    save_path: str,
    theme_names=None,
) -> None:
    if metric not in df.columns or df[metric].isna().all():
        return

    networks = sorted(df["network"].unique().tolist(), key=lambda s: s.lower())
    algorithms = _sort_algorithms(df["algorithm"].unique())
    x = np.arange(len(networks), dtype=float)
    n_algos = max(len(algorithms), 1)
    width = 0.82 / n_algos

    for theme in resolve_themes(theme_names):
        styles = build_algorithm_styles(algorithms, theme)
        with theme_context(theme):
            fig, ax = plt.subplots(figsize=(9.2, 5.2))
            for i, alg in enumerate(algorithms):
                sub = df[df["algorithm"] == alg].set_index("network")
                values = np.array(
                    [sub[metric].get(net, np.nan) for net in networks], dtype=float
                )
                errs = np.array(
                    [
                        sub.get(f"{metric}_std", pd.Series()).get(net, 0.0)
                        for net in networks
                    ],
                    dtype=float,
                )
                color, _ = styles[alg]
                offset = (i - (n_algos - 1) / 2.0) * width
                ax.bar(
                    x + offset,
                    np.nan_to_num(values, nan=0.0),
                    width=width * 0.94,
                    color=color,
                    label=alg,
                    yerr=np.nan_to_num(errs, nan=0.0),
                    capsize=3.0,
                    edgecolor=theme.axes_face,
                    linewidth=0.8,
                    zorder=3,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(networks)
            ax.set_ylabel(METRIC_METADATA[metric]["label"], labelpad=10)
            ax.set_axisbelow(True)
            ax.grid(axis="y", which="major", linewidth=0.9, color=theme.grid_major)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color(theme.spine)
            ax.spines["bottom"].set_color(theme.spine)
            if metric == "time":
                ax.set_yscale("log")

            legend = ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.12),
                ncol=min(n_algos, 4),
                columnspacing=1.2,
                handlelength=1.8,
            )
            _legend_frame(legend, theme)
            fig.tight_layout()
            export_figure(fig, save_path, f"real_nets_{metric}", theme)


def main() -> None:
    df = run_all(n_runs=N_RUNS)
    out_dir = os.path.join(SAVE_PATH, "real_nets") + "/"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "real_nets.csv")
    df.to_csv(csv_path, index=False)
    print("\n" + df.to_string(index=False))
    for metric in ("modularity", "nmi", "ami", "time"):
        _bar_plot(df, metric, out_dir)
    print(f"\nReal-network benchmark saved to {out_dir}")


if __name__ == "__main__":
    main()

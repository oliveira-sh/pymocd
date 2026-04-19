import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from typing import Dict, Any, Iterable, Sequence

from utils import generate_lfr_benchmark, evaluate_communities, plot_results, SAVE_PATH
from .registry import ALGORITHM_REGISTRY
from .constants import MIN_MU, MAX_MU, STP_MU, NUM_RUNS


class ExperimentRunner:
    def __init__(self, n_runs: int = NUM_RUNS):
        self.n_runs = n_runs

    def _run_single(self, args: tuple) -> Dict[str, Any]:
        (
            alg_name,
            alg_func,
            needs_conversion,
            n_runs,
            mu,
            n_nodes,
            backup_csv,
        ) = args
        metrics = {"modularity": [], "nmi": [], "ami": [], "time": []}

        for run_id in tqdm(
            range(n_runs), desc=f"{alg_name}", leave=False, disable=n_runs == 1
        ):
            G, ground_truth = generate_lfr_benchmark(
                n=int(n_nodes), mu=float(mu), seed=run_id
            )
            start = time.time()
            communities = alg_func(G, seed=run_id)
            duration = time.time() - start

            eval_result = evaluate_communities(
                G, communities, ground_truth, convert=needs_conversion
            )
            metrics["modularity"].append(eval_result["modularity"])
            metrics["nmi"].append(eval_result["nmi"])
            metrics["ami"].append(eval_result["ami"])
            metrics["time"].append(duration)

        result = {
            "algorithm": alg_name,
            "mu": float(mu),
            "nodes": int(n_nodes),
        }
        for m in metrics:
            result[f"{m}_mean"] = float(np.mean(metrics[m]))
            result[f"{m}_std"] = float(
                np.std(metrics[m], ddof=min(1, len(metrics[m]) - 1))
            )

        try:
            os.makedirs(SAVE_PATH, exist_ok=True)
            row = pd.DataFrame([result]).rename(
                columns={
                    "modularity_mean": "modularity",
                    "nmi_mean": "nmi",
                    "ami_mean": "ami",
                    "time_mean": "time",
                }
            )
            row.to_csv(
                backup_csv, mode="a", header=not os.path.exists(backup_csv), index=False
            )
        except Exception:
            pass

        return result

    def _build_args(
        self,
        mus: Sequence[float],
        n_nodes_list: Sequence[int],
        backup_csv: str,
    ):
        parallel_args, sequential_args = [], []
        for name, info in ALGORITHM_REGISTRY.items():
            for mu in mus:
                for n in n_nodes_list:
                    entry = (
                        name,
                        info["function"],
                        info["needs_conversion"],
                        self.n_runs,
                        float(mu),
                        int(n),
                        backup_csv,
                    )
                    (parallel_args if info["parallel"] else sequential_args).append(
                        entry
                    )
        return parallel_args, sequential_args

    def _run_sweep(
        self,
        mus: Sequence[float],
        n_nodes_list: Sequence[int],
        csv_file: str,
        plot_subdir: str,
    ) -> pd.DataFrame:
        backup_csv = csv_file.replace(".csv", "_bk.csv")
        parallel_args, sequential_args = self._build_args(mus, n_nodes_list, backup_csv)
        total = len(parallel_args) + len(sequential_args)

        results = []
        with tqdm(total=total, desc="Benchmarking", unit="task") as pbar:
            if parallel_args:
                with Pool() as pool:
                    for result in pool.imap_unordered(self._run_single, parallel_args):
                        results.append(result)
                        pbar.set_postfix(
                            alg=result["algorithm"],
                            mu=f"{result['mu']:.2f}",
                            n=result["nodes"],
                        )
                        pbar.update()
            for args in sequential_args:
                result = self._run_single(args)
                results.append(result)
                pbar.set_postfix(
                    alg=result["algorithm"],
                    mu=f"{result['mu']:.2f}",
                    n=result["nodes"],
                )
                pbar.update()

        df = pd.DataFrame(results).rename(
            columns={
                "modularity_mean": "modularity",
                "nmi_mean": "nmi",
                "ami_mean": "ami",
                "time_mean": "time",
            }
        )
        os.makedirs(SAVE_PATH, exist_ok=True)
        df.to_csv(csv_file, index=False)
        plot_results(df, save_path=plot_subdir)
        return df

    def run_mu_experiment(
        self,
        mus: Iterable[float] | None = None,
        n_nodes_list: Iterable[int] | None = None,
    ) -> pd.DataFrame:
        if mus is None:
            mus = np.arange(MIN_MU, MAX_MU + STP_MU, STP_MU)
        if n_nodes_list is None:
            n_nodes_list = [500, 1000, 2000]
        csv_file = os.path.join(SAVE_PATH, "lfr_mu.csv")
        plot_subdir = os.path.join(SAVE_PATH, "mu") + "/"
        return self._run_sweep(list(mus), list(n_nodes_list), csv_file, plot_subdir)

    def run_nodes_experiment(
        self,
        n_list: Iterable[int] | None = None,
        mus: Iterable[float] | None = None,
    ) -> pd.DataFrame:
        if n_list is None:
            n_list = [500, 1000, 2000, 5000]
        if mus is None:
            mus = [0.3, 0.5]
        csv_file = os.path.join(SAVE_PATH, "lfr_nodes.csv")
        plot_subdir = os.path.join(SAVE_PATH, "nodes") + "/"
        return self._run_sweep(list(mus), list(n_list), csv_file, plot_subdir)

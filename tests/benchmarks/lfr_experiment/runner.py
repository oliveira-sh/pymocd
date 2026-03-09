import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from typing import Dict, Any

from utils import generate_lfr_benchmark, evaluate_communities, plot_results, SAVE_PATH
from .registry import ALGORITHM_REGISTRY
from .algorithms import MIN_MU, MAX_MU, STP_MU, NUM_RUNS


class ExperimentRunner:
    def __init__(self, n_runs: int = NUM_RUNS):
        self.n_runs = n_runs

    def _run_single(self, args: tuple) -> Dict[str, Any]:
        (
            alg_name,
            alg_func,
            needs_conversion,
            n_runs,
            param_name,
            param_value,
            fixed_param,
            backup_csv,
        ) = args
        metrics = {"modularity": [], "nmi": [], "ami": [], "time": []}

        for run_id in tqdm(
            range(n_runs), desc=f"{alg_name}", leave=False, disable=n_runs == 1
        ):
            if param_name == "mu":
                G, ground_truth = generate_lfr_benchmark(
                    n=fixed_param, mu=param_value, seed=run_id
                )
            else:
                G, ground_truth = generate_lfr_benchmark(
                    n=param_value, mu=fixed_param, seed=run_id
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

        result = {"algorithm": alg_name, param_name: param_value}
        for m in metrics:
            result[f"{m}_mean"] = np.mean(metrics[m])
            result[f"{m}_std"] = np.std(metrics[m], ddof=min(1, len(metrics[m]) - 1))

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

    def _build_args(self, param_name, param_values, fixed_param, backup_csv):
        parallel_args, sequential_args = [], []
        for name, info in ALGORITHM_REGISTRY.items():
            for val in param_values:
                entry = (
                    name,
                    info["function"],
                    info["needs_conversion"],
                    self.n_runs,
                    param_name,
                    val,
                    fixed_param,
                    backup_csv,
                )
                (parallel_args if info["parallel"] else sequential_args).append(entry)
        return parallel_args, sequential_args

    def _run(self, param_name, param_values, fixed_param, csv_file, plot_subdir):
        backup_csv = csv_file.replace(".csv", "_bk.csv")
        parallel_args, sequential_args = self._build_args(
            param_name, param_values, fixed_param, backup_csv
        )
        total = len(parallel_args) + len(sequential_args)

        results = []
        with tqdm(total=total, desc="Benchmarking", unit="task") as pbar:
            if parallel_args:
                with Pool() as pool:
                    for result in pool.imap_unordered(self._run_single, parallel_args):
                        results.append(result)
                        pbar.set_postfix(
                            alg=result["algorithm"], **{param_name: result[param_name]}
                        )
                        pbar.update()
            for args in sequential_args:
                result = self._run_single(args)
                results.append(result)
                pbar.set_postfix(
                    alg=result["algorithm"], **{param_name: result[param_name]}
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

    def run_mu_experiment(self, mus=None, n_nodes=250):
        if mus is None:
            mus = np.arange(MIN_MU, MAX_MU + STP_MU, STP_MU)
        csv_file = os.path.join(SAVE_PATH, "lfr_mu.csv")
        plot_subdir = os.path.join(SAVE_PATH, "mu") + "/"
        return self._run("mu", mus, n_nodes, csv_file, plot_subdir)

    def run_nodes_experiment(self, n_list=None, mu=0.3):
        if n_list is None:
            n_list = np.arange(10_000, 60_000, 10_000)
        csv_file = os.path.join(SAVE_PATH, "lfr_nodes.csv")
        plot_subdir = os.path.join(SAVE_PATH, "nodes") + "/"
        return self._run("nodes", n_list, mu, csv_file, plot_subdir)

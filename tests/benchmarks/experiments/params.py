import os
import time
import pymocd
import networkx as nx
import pandas as pd

from ..core.utils import generate_lfr_benchmark, evaluate_communities

network_sizes    = [10000, 20000, 30000]
population_sizes = [50, 100, 150, 200]
generation_counts= [50 * i for i in range(1, 7)]
n_runs           = 20

output_file = 'ga_params_experiment_results.csv'
if os.path.exists(output_file):
    os.remove(output_file)

for n in network_sizes:
    for pop_size in population_sizes:
        for num_gens in generation_counts:
            for run_id in range(1, n_runs + 1):
                G, ground_truth = generate_lfr_benchmark(
                    n=n,
                    seed=run_id + num_gens - pop_size
                )

                start = time.time()
                solver = pymocd.HpMocd(
                    G,
                    debug_level=0,
                    pop_size=pop_size,
                    num_gens=num_gens
                )
                rdict = solver.run()
                elapsed = time.time() - start

                metrics = evaluate_communities(
                    G, rdict, ground_truth,
                    convert=False
                )

                row = {
                    'n':           n,
                    'pop_size':    pop_size,
                    'num_gens':    num_gens,
                    'run_id':      run_id,
                    'modularity':  float(metrics['modularity']),
                    'nmi':         float(metrics['nmi']),
                    'ami':         float(metrics['ami']),
                    'runtime_sec': elapsed
                }
                df_row = pd.DataFrame([row])

                df_row.to_csv(
                    output_file,
                    mode='a',
                    header=not os.path.exists(output_file),
                    index=False
                )

                print(f"[n={n}] Saved run {run_id} (pop={pop_size}, gens={num_gens})")

print("All runs complete. Results in", output_file)

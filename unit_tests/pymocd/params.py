import time
import pymocd
import networkx as nx
import pandas as pd

from utils import generate_lfr_benchmark, evaluate_communities

population_sizes = [50, 100, 150, 200]
generation_counts = [50 * i for i in range(1, 11)]
# independent trials per configuration
n_runs = 20 

records = []

for pop_size in population_sizes:
    for num_gens in generation_counts:
        for run_id in range(1, n_runs + 1):
            G, ground_truth = generate_lfr_benchmark(seed=run_id + num_gens - pop_size)
            
            start = time.time()
            solver = pymocd.HpMocd(G, debug_level=3, pop_size=pop_size, num_gens=num_gens)
            rdict = solver.run()
            elapsed = time.time() - start
            
            metrics = evaluate_communities(G, rdict, ground_truth, convert=False)
            
            records.append({
                'pop_size':    pop_size,
                'num_gens':    num_gens,
                'run_id':      run_id,
                'modularity':  float(metrics['modularity']),
                'nmi':         float(metrics['nmi']),
                'ami':         float(metrics['ami']),
                'runtime_sec': elapsed
            })

df = pd.DataFrame(records)
df.to_csv('ga_params_experiment_results.csv', index=False)
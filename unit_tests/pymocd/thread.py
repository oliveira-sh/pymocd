import os
import time
import matplotlib.pyplot as plt
import pymocd
from utils import generate_lfr_benchmark, SAVE_PATH

num_threads    = 8
runs_per_setting = 5
os.makedirs(SAVE_PATH, exist_ok=True)

G, _ = generate_lfr_benchmark()
avg_times = []

pymocd.set_thread_count(num_threads)
    
_ = pymocd.HpMocd(G).run()
    
times = []
for i in range(runs_per_setting):
        model = pymocd.HpMocd(G, debug_level=3)
        start = time.perf_counter()
        partition = model.run()
        end   = time.perf_counter()
        times.append(end - start)
    
avg = sum(times) / runs_per_setting
avg_times.append(avg)
print(f"threads={num_threads:2} → runs: {['{:.3f}s'.format(t) for t in times]}  avg={avg:.3f}s")


# threads= 2 → runs: ['10.545s', '10.941s', '11.201s', '10.701s', '10.629s']  avg=10.804s
# threads= 4 → runs: ['6.238s', '5.971s', '6.299s', '6.297s', '6.170s']  avg=6.195s
# threads= 6 → runs: ['4.925s', '5.031s', '5.178s', '4.987s', '4.927s']  avg=5.010
# threads= 8 → runs: ['4.579s', '4.555s', '4.703s', '4.810s', '4.576s']  avg=4.645s
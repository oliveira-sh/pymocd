#!/usr/bin/env python3
import sys
import os
import csv
import time

import pymocd
from utils import generate_lfr_benchmark, SAVE_PATH

def main():
    # 1) parse command line
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} NUM_THREADS")
        sys.exit(1)
    num_threads = int(sys.argv[1])

    runs_per_setting = 5
    os.makedirs(SAVE_PATH, exist_ok=True)
    csv_file = os.path.join(SAVE_PATH, 'threads_benchmark.csv')

    G, _ = generate_lfr_benchmark()

    # 4) set threads & run once to “warm up” if desired
    pymocd.set_thread_count(num_threads)
    _ = pymocd.HpMocd(G).run()

    # 5) do timed runs
    times = []
    for i in range(runs_per_setting):
        model = pymocd.HpMocd(G, debug_level=3)
        start = time.perf_counter()
        _ = model.run()
        end = time.perf_counter()
        elapsed = end - start
        times.append(elapsed)

    # 6) compute average
    avg = sum(times) / runs_per_setting

    # 7) print summary
    formatted = [f"{t:.3f}s" for t in times]
    print(f"threads={num_threads:2} → runs: {formatted}  avg={avg:.3f}s")

    # 8) append to CSV (create + header if needed)
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ['threads'] \
                   + [f'run{i+1}' for i in range(runs_per_setting)] \
                   + ['avg']
            writer.writerow(header)
        # write raw floats for easier plotting later
        writer.writerow([num_threads]
                        + [f"{t:.6f}" for t in times]
                        + [f"{avg:.6f}"])

if __name__ == '__main__':
    main()

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import time

import pymocd

from utils import generate_lfr_benchmark, SAVE_PATH

RUNS_PER_SETTING = 5


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python3 {sys.argv[0]} NUM_THREADS")
        sys.exit(1)
    num_threads = int(sys.argv[1])

    os.makedirs(SAVE_PATH, exist_ok=True)
    csv_file = os.path.join(SAVE_PATH, "threads_benchmark.csv")

    graph, _ = generate_lfr_benchmark()

    pymocd.max_cores(num_threads)
    pymocd.hpmocd(graph)

    times = []
    for _ in range(RUNS_PER_SETTING):
        start = time.perf_counter()
        pymocd.hpmocd(graph)
        times.append(time.perf_counter() - start)

    avg = sum(times) / RUNS_PER_SETTING

    formatted = [f"{t:.3f}s" for t in times]
    print(f"threads={num_threads:2} -> runs: {formatted}  avg={avg:.3f}s")

    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["threads"] + [f"run{i + 1}" for i in range(RUNS_PER_SETTING)] + ["avg"]
            )
        writer.writerow([num_threads] + [f"{t:.6f}" for t in times] + [f"{avg:.6f}"])


if __name__ == "__main__":
    main()

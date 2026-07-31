import os

MIN_MU = 0.1
MAX_MU = 0.7
STP_MU = 0.1
NUM_ND = 1_000

NUM_RUNS = int(os.environ.get("PYMOCD_BENCH_RUNS", "2"))
DEBUG = True

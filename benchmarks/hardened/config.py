import os

SMOKE = bool(os.environ.get("HARD_SMOKE"))

RUNS = 2 if SMOKE else int(os.environ.get("HARD_RUNS", "20"))
TIMEOUT = int(os.environ.get("HARD_TIMEOUT", "60" if SMOKE else "43200"))
THREADS = int(os.environ.get("HARD_THREADS", "2" if SMOKE else "12"))
WORKERS = int(os.environ.get("HARD_WORKERS", "2" if SMOKE else "4"))
EXCLUSIVE_N = int(os.environ.get("HARD_EXCLUSIVE_N", "250000"))
EXCLUSIVE_THREADS = int(os.environ.get("HARD_EXCLUSIVE_THREADS", "48"))

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
LFR_DIR = os.path.join(DATA, "lfr")
OUT = os.path.join(os.path.dirname(HERE), "results", "hardened")
RESULTS_CSV = os.path.join(OUT, "results.csv")

LFR_PARAMS = dict(tau1=2.5, tau2=1.5, average_degree=20, max_degree=50,
                  min_community=20, max_community=100)

if SMOKE:
    MU_SWEEP_N = [300]
    MU_SWEEP_MU = [0.3, 0.5]
    NODES_SWEEP_N = [300, 600]
    NODES_SWEEP_MU = [0.3]
else:
    MU_SWEEP_N = [50_000, 100_000]
    MU_SWEEP_MU = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    NODES_SWEEP_N = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
    NODES_SWEEP_MU = [0.3, 0.5]

REAL_SMALL = ["karate", "dolphins", "lesmis", "florentine", "polbooks",
              "football", "email_eu"]
REAL_SNAP = [] if SMOKE else ["dblp", "amazon", "youtube", "lj", "orkut"]

# deterministic algorithms get one run per fixed real graph (exact); on LFR
# every seed is a different graph, so RUNS runs happen regardless.
ALGORITHMS = {
    "SMOCC":       dict(deterministic=True,  max_nodes=None,      needs="shim"),
    "HP-MOCD":     dict(deterministic=False, max_nodes=None,      needs="shim"),
    "MMCoMO":      dict(deterministic=False, max_nodes=2_000,     needs="shim"),
    "NSGA-III CCM": dict(deterministic=False, max_nodes=None,     needs="shim"),
    "NSGA-III KRM": dict(deterministic=False, max_nodes=None,     needs="shim"),
    "Shi-MOCD (Q)": dict(deterministic=False, max_nodes=None,     needs="shim"),
    "Shi-MOCD (D)": dict(deterministic=False, max_nodes=None,     needs="shim"),
    "MOGA-Net":    dict(deterministic=False, max_nodes=None,      needs="shim"),
    "Louvain":     dict(deterministic=False, max_nodes=2_000_000, needs="nx"),
    "Leiden":      dict(deterministic=False, max_nodes=None,      needs="ig"),
    "ASYN-LPA":    dict(deterministic=False, max_nodes=2_000_000, needs="nx"),
}

CSV_FIELDS = ["alg", "kind", "net", "n_cfg", "mu", "seed", "status", "n", "m",
              "k", "time", "nmi", "ami", "modularity", "threads", "stamp"]


def task_key(t):
    return (t["alg"], t["kind"], t.get("net", ""), t.get("n_cfg", ""),
            t.get("mu", ""), t["seed"])


def row_key(r):
    def norm(v):
        s = str(v)
        if s in ("", "nan"):
            return ""
        try:
            f = float(s)
            return str(int(f)) if f == int(f) else str(f)
        except ValueError:
            return s
    return (r["alg"], r["kind"], norm(r["net"]), norm(r["n_cfg"]),
            norm(r["mu"]), norm(r["seed"]))


def norm_task_key(t):
    return row_key({"alg": t["alg"], "kind": t["kind"],
                    "net": t.get("net", ""), "n_cfg": t.get("n_cfg", ""),
                    "mu": t.get("mu", ""), "seed": t["seed"]})

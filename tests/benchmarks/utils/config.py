import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import rcParams

_run_id = os.environ.get("BENCHMARK_RUN_ID") or datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)
SAVE_PATH = (
    str(
        Path(__file__).resolve().parent.parent.parent.parent
        / "tests"
        / "outputs"
        / _run_id
    )
    + "/"
)

plt.style.use("seaborn-v0_8-whitegrid")
rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 18,
        "text.usetex": False,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "patch.linewidth": 1.0,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

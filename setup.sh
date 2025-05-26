#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [OPTION]

Options:
  --dependencies      Create venv, install Python deps & build Rust extension
  --experiment        Run LFR experiment (after setting up dependencies)
  --pareto-frontier   Run Pareto-front experiment (after setting up dependencies)
  -h, --help          Show this help message
EOF
}

setup() {# Register with parallel flags as desired

  echo "[..] Creating virtual environment at .venv…"
  python3 -m venv .venv

  # shellcheck source=/dev/null
  source .venv/bin/activate

  echo "[..] Upgrading pip, setuptools, and wheel…"
  pip install --upgrade pip setuptools wheel

  echo "[..] Installing required Python packages…"
  pip install \
    networkx \
    maturin \
    pandas \
    matplotlib \
    scikit-learn \
    tqdm \
    numpy \
    python-louvain \
    igraph \
    leidenalg \
    pymoo

  echo "[..] Building the Rust extension via maturin…"
  maturin develop --release

  echo "[OK] Setup complete."
}

case "${1:-}" in
  --dependencies)
    setup
    ;;
  --experiment)
    setup
    echo "[..] Running LFR experiment and writing to experiment.out…"
    python3 unit_tests/pymocd/lfr_experiment.py > experiment.out
    echo "[OK] Experiment finished. See experiment.out."
    ;;
  --pareto-frontier)
    setup
    echo "[..] Running Pareto-front experiment and writing to experiment.out…"
    python3 unit_tests/pymocd/pareto_front.py > experiment.out
    echo "[OK] Experiment finished. See experiment.out."
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "Error: unknown option '$1'"
    usage
    exit 1
    ;;
esac

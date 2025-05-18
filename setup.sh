#!/bin/bash

set -e 
set -o pipefail

echo "[..] Creating virtual environment..."
python3 -m venv .venv

echo "[OK] Virtual environment created at .venv"

source .venv/bin/activate

echo "[..] Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo "[..] Installing required Python packages..."
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
    pymoo \

maturin develop --release

echo "[OK] Setup complete"

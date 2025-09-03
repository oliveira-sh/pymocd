# PyMOCD Benchmarks

This directory contains benchmarking tools and experiments for evaluating multi-objective community detection algorithms.

## Directory Structure

```
benchmarks/
├── core/                    # Core utilities and shared functions
│   ├── __init__.py
│   └── utils.py            # Benchmark generation, evaluation, and plotting utilities
├── experiments/            # Benchmark experiments
│   ├── __init__.py
│   ├── lfr_experiment.py   # Comprehensive algorithm comparison on LFR networks
│   └── params.py           # Genetic algorithm parameter tuning experiments
├── visualization/          # Visualization and plotting tools
│   ├── __init__.py
│   ├── evolutionary.py     # Community evolution visualization across generations
│   ├── pareto_front.py     # Pareto frontier analysis and visualization
│   └── plot.py             # Simple plotting utility for CSV results
└── README.md               # This file
```

## Usage

### Running Experiments

1. **LFR Algorithm Comparison**:
   ```bash
   cd experiments/
   python -m lfr_experiment
   ```

2. **Parameter Tuning**:
   ```bash
   cd experiments/
   python -m params
   ```

### Generating Visualizations

1. **Evolutionary Visualization**:
   ```bash
   cd visualization/
   python -m evolutionary
   ```

2. **Pareto Front Analysis**:
   ```bash
   cd visualization/
   python -m pareto_front
   ```

3. **Plot Existing Results**:
   ```bash
   cd visualization/
   python -m plot path/to/results.csv
   ```

## Environment Setup

Make sure to activate the virtual environment:
```bash
source ../../.venv/bin/activate
```

## Output

Results and plots are saved to `tests/output/` directory by default.
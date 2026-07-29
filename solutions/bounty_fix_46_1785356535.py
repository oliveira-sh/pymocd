repository_root/
├── pyproject.toml              # Pytest configuration & markers
├── src/
│   └── algorithm_lib/
│       ├── __init__.py
│       └── sorting.py          # Example algorithms under test
└── tests/
    ├── conftest.py             # Pytest hooks, options, and fixtures
    ├── unit/
    │   ├── __init__.py
    │   └── test_sorting_unit.py # Correctness & standard functionality
    └── benchmarks/
        ├── __init__.py
        ├── benchmark_utils.py  # Statistical benchmarking toolset
        └── test_sorting_bench.py # Comparative benchmarks (e.g., QuickSort vs MergeSort)
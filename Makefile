.PHONY: all dependencies stubs build test benchmark benchmark-params benchmark-synthetic benchmark-real benchmark-parallelism hard-gen hard-run hard-report hard-smoke bump clean docs docs-serve

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
BENCH_PY := $(CURDIR)/$(PYTHON)
BENCH_THREADS := 1 2 4 8

ifndef BENCHMARK_RUN_ID
BENCHMARK_RUN_ID := $(shell date +%Y-%m-%d_%H-%M-%S)
endif

all: build test

$(VENV)/bin/activate: res/requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r res/requirements.txt
	@touch $(VENV)/bin/activate

dependencies: $(VENV)/bin/activate

stubs:
	cargo run --bin stub_gen

build: dependencies
	cargo run --bin stub_gen
	$(VENV)/bin/maturin develop --release
	@rm -f pymocd.pyi

test: build
	cargo test --manifest-path=Cargo.toml

benchmark-params: build
	cd benchmarks && BENCHMARK_RUN_ID=$(BENCHMARK_RUN_ID) $(BENCH_PY) -m alg_params.num_generations
	cd benchmarks && BENCHMARK_RUN_ID=$(BENCHMARK_RUN_ID) $(BENCH_PY) -m alg_params.pareto_front
	cd benchmarks && BENCHMARK_RUN_ID=$(BENCHMARK_RUN_ID) $(BENCH_PY) -m alg_params.population_size

benchmark-synthetic: build
	cd benchmarks && BENCHMARK_RUN_ID=$(BENCHMARK_RUN_ID) $(BENCH_PY) -m _exp_synt_net

benchmark-real: build
	cd benchmarks && BENCHMARK_RUN_ID=$(BENCHMARK_RUN_ID) $(BENCH_PY) -m _exp_real_net

benchmark-parallelism: build
	cd benchmarks && for threads in $(BENCH_THREADS); do \
		BENCHMARK_RUN_ID=$(BENCHMARK_RUN_ID) $(BENCH_PY) -m parallelism.thread_scaling $$threads; \
	done

benchmark: benchmark-params benchmark-synthetic benchmark-real benchmark-parallelism
	@echo "Results saved to benchmarks/results/$(BENCHMARK_RUN_ID)/"

docs: dependencies stubs
	$(PIP) install -q -r docs/requirements.txt
	$(VENV)/bin/mkdocs build

docs-serve: dependencies stubs
	$(PIP) install -q -r docs/requirements.txt
	$(VENV)/bin/mkdocs serve

bump:
	$(if $(V),,$(error Usage: make bump V=2.0.2))
	sed -i 's/^version = ".*"/version = "$(V)"/' Cargo.toml pyproject.toml
	git add Cargo.toml pyproject.toml
	git commit -m "chore: bump version to $(V)"
	git tag v$(V)
	git push origin master --follow-tags

clean:
	@cargo clean
	@rm -rf $(VENV) target build dist *.egg-info pymocd.pyi
	@find . -type d -name "__pycache__" -exec rm -rf {} +

# --- hardened campaign (Threadripper): resumable, incremental, 12h timeout ---
hard-gen: build
	cd benchmarks && $(BENCH_PY) hardened/gen_graphs.py

hard-run: build
	cd benchmarks && $(BENCH_PY) hardened/run.py

hard-report:
	cd benchmarks/hardened && $(BENCH_PY) report.py $(METRIC)

hard-smoke: build
	cd benchmarks && HARD_SMOKE=1 $(BENCH_PY) hardened/run.py

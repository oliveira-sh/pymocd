.PHONY: all dependencies build test benchmark bump clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

all: build test

$(VENV)/bin/activate: res/requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r res/requirements.txt
	@touch $(VENV)/bin/activate

dependencies: $(VENV)/bin/activate

build: dependencies
	$(VENV)/bin/maturin develop --release

test: build
	cargo test --manifest-path=Cargo.toml

benchmark: build
	$(eval BENCHMARK_RUN_ID := $(shell date +%Y-%m-%d_%H-%M-%S))
	#cd tests/benchmarks && BENCHMARK_RUN_ID=$(BENCHMARK_RUN_ID) $(CURDIR)/$(PYTHON) evolutionary.py
	#cd tests/benchmarks && BENCHMARK_RUN_ID=$(BENCHMARK_RUN_ID) $(CURDIR)/$(PYTHON) pareto_front.py
	cd tests/benchmarks && BENCHMARK_RUN_ID=$(BENCHMARK_RUN_ID) $(CURDIR)/$(PYTHON) -m lfr_experiment
	@echo "Results saved to tests/outputs/$(BENCHMARK_RUN_ID)/"

bump:
	$(if $(V),,$(error Usage: make bump V=2.0.2))
	sed -i 's/^version = ".*"/version = "$(V)"/' Cargo.toml pyproject.toml
	git add Cargo.toml pyproject.toml
	git commit -m "chore: bump version to $(V)"
	git tag v$(V)
	git push origin master --follow-tags

clean:
	@cargo clean
	@rm -rf $(VENV) target build dist *.egg-info
	@find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: all dependencies build test benchmark clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

all: build test

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate

dependencies: $(VENV)/bin/activate

build: dependencies
	$(VENV)/bin/maturin develop --release

test: build
	cargo test --manifest-path=Cargo.toml

benchmark: build
	cd tests/benchmarks && $(CURDIR)/$(PYTHON) evolutionary.py
	cd tests/benchmarks && $(CURDIR)/$(PYTHON) pareto_front.py
	cd tests/benchmarks && $(CURDIR)/$(PYTHON) lfr_experiment.py

clean:
	@cargo clean
	@rm -rf $(VENV) target build dist *.egg-info
	@find . -type d -name "__pycache__" -exec rm -rf {} +

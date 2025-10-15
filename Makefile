.PHONY: setup build test clean

setup:
	@echo "Setting up development environment..."
	@if [ -d "venv" ]; then \
		echo "Removing existing venv..."; \
		rm -rf venv; \
	fi
	@python -m venv venv
	@./venv/bin/pip install --upgrade pip
	@./venv/bin/pip install \
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
		pybind11 \
		walker \
		seaborn
	@echo "Setup complete. Activate with: source venv/bin/activate"

build:
	@echo "Building package..."
	@if [ -d "venv" ]; then \
		if ! ./venv/bin/python -c "import maturin" 2>/dev/null; then \
			./venv/bin/pip install maturin; \
		fi; \
		./venv/bin/maturin develop --release; \
	else \
		if ! command -v maturin >/dev/null 2>&1; then \
			pip install maturin; \
		fi; \
		maturin develop --release; \
	fi
	@echo "Build complete."

test: build
	@echo "Testing package..."
	@if [ -d "venv" ]; then \
		./venv/bin/python -c "import pymocd; print('Package import: OK')"; \
		./venv/bin/python -c "import pymocd; print('hello_world():', pymocd.hello_world())"; \
		./venv/bin/python -c "import pymocd; print('Functions available:', len([x for x in dir(pymocd) if not x.startswith('_')]))"; \
	else \
		python -c "import pymocd; print('Package import: OK')"; \
		python -c "import pymocd; print('hello_world():', pymocd.hello_world())"; \
		python -c "import pymocd; print('Functions available:', len([x for x in dir(pymocd) if not x.startswith('_')]))"; \
	fi
	@cargo test --manifest-path=Cargo.toml
	@echo "Tests complete."

# Clean build artifacts
clean:
	@echo "Cleaning..."
	@cargo clean
	@rm -rf target/
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info/
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete."
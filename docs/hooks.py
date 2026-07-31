"""Expose the generated pyo3 stub to mkdocstrings.

API pages are rendered from ``pymocd.pyi`` (generated from the Rust sources
by ``make stubs``), so the docs always match the compiled library. griffe
only loads ``.py`` search-path modules, hence the shadow copy.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def on_startup(command, dirty):
    stub = ROOT / "pymocd.pyi"
    if not stub.exists():
        raise FileNotFoundError("pymocd.pyi not found — run `make stubs` first")
    dest = ROOT / ".docs-stub"
    dest.mkdir(exist_ok=True)
    shutil.copy(stub, dest / "pymocd.py")

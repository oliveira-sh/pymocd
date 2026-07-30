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

    assets = ROOT / "docs" / "assets"
    assets.mkdir(exist_ok=True)
    for svg in ("logo.svg", "icon.svg"):
        shutil.copy(ROOT / "res" / svg, assets / svg)

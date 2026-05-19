"""
plotting.py — Shared style helpers for every notebook.

Re-exports from snp.py so notebooks can `from scripts.plotting import ...`
without dragging in the full SNP data class.
"""
from pathlib import Path
from scripts.snp import (
    PALETTE,
    _apply_style as apply_style,
    title,
    _rel as rel_path,
)

_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = _ROOT / "plots"


def fig_path(subdir: str, name: str, ext: str = "pdf") -> Path:
    """Return plots/<subdir>/<name>.<ext>, creating the directory."""
    d = PLOTS_DIR / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}.{ext}"


__all__ = ["PALETTE", "apply_style", "title", "rel_path", "PLOTS_DIR", "fig_path"]

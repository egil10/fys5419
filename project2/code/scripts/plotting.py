"""
plotting.py — Shared style helpers for every notebook.

Re-exports from snp.py so notebooks can `from scripts.plotting import ...`
without dragging in the full SNP data class. `fig_path()` routes to
`scripts.colab.out_dir('plots', subdir)`, which lives in Google Drive on
Colab (so PDFs survive runtime restarts) and in the local repo otherwise.
"""
from pathlib import Path
from scripts.snp   import (
    PALETTE,
    _apply_style as apply_style,
    title,
    _rel as rel_path,
)
from scripts.colab import out_dir


def fig_path(subdir: str, name: str, ext: str = "pdf") -> Path:
    """Return `<plots>/<subdir>/<name>.<ext>`, creating the directory.

    On Colab `<plots>` is `/content/drive/MyDrive/GITHUB-COLAB/fys5419/
    project2/code/plots`; locally it is `<repo>/project2/code/plots`.
    """
    return out_dir("plots", subdir) / f"{name}.{ext}"


__all__ = ["PALETTE", "apply_style", "title", "rel_path", "fig_path", "out_dir"]

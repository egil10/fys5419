"""
exports.py — Bundle every script and notebook into .md, .html, and .pdf.

Drop into `project2/code/.exports/` so a report-writing agent can pull a
single bundle of human-readable artefacts without poking around the repo:

    .exports/
        scripts/
            qaoa.md  qaoa.html  qaoa.pdf
            ...
        notebooks/
            03_qaoa.md  03_qaoa.html  03_qaoa.pdf
            ...

Conversions
-----------
- `.py`  -> .md  : wrap source in a fenced ```python block.
- `.py`  -> .html: pygments HtmlFormatter (full HTML page).
- `.py`  -> .pdf : tiny `\\lstinputlisting` LaTeX wrapper + pdflatex.
- `.ipynb` -> .md / .html / .pdf : nbconvert (xelatex backend for PDF).

PDF generation needs `pdflatex` (LaTeX) on PATH for both code and notebook
output. On Windows that's typically TinyTeX, MiKTeX, or TeX Live. If
LaTeX is missing, the script skips PDF and still produces .md / .html.

CLI
---
    python -m scripts.exports                 # everything to .exports/
    python -m scripts.exports --only md html  # skip PDF
    python -m scripts.exports --out my_dir    # custom output root

Idempotent: re-running overwrites existing files. Safe to invoke at the
end of the pipeline as a final "package up for the report" step.
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent           # project2/code
SCRIPTS   = ROOT / "scripts"
NOTEBOOKS = ROOT / "notebooks"
DEFAULT_OUT = ROOT / ".exports"


# ── Code <-> Markdown / HTML / PDF ───────────────────────────────────────
def _rel(src: Path) -> str:
    """Path of `src` relative to the repo root for display, fall back to name."""
    src = src.resolve()
    try:
        return src.relative_to(ROOT.parent.parent).as_posix()  # repo root
    except ValueError:
        try:
            return src.relative_to(ROOT.parent).as_posix()
        except ValueError:
            return src.name


def py_to_md(src: Path, dst: Path) -> None:
    """Wrap a .py file in a fenced Python block, add a path-aware header."""
    body = src.read_text(encoding="utf-8")
    dst.write_text(f"# `{_rel(src)}`\n\n```python\n{body}\n```\n",
                   encoding="utf-8")


def py_to_html(src: Path, dst: Path) -> None:
    """Render the .py with pygments as a self-contained HTML page."""
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter

    body = src.read_text(encoding="utf-8")
    fmt = HtmlFormatter(full=True, style="friendly",
                        title=_rel(src), linenos="table")
    dst.write_text(highlight(body, PythonLexer(), fmt), encoding="utf-8")


def py_to_pdf(src: Path, dst: Path) -> bool:
    """Wrap the .py in a minimal listings document and shell out to pdflatex.

    Returns True on success, False if pdflatex is missing or the build
    fails — caller can fall back to skipping PDFs without crashing.
    """
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        return False

    src = src.resolve()
    # Escape underscores in the path for the LaTeX title line.
    safe_rel = _rel(src).replace("_", r"\_")

    tex_src = rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[margin=2cm]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{listings}}
\usepackage{{xcolor}}
\definecolor{{codegrey}}{{rgb}}{{0.95,0.95,0.95}}
\lstset{{
  basicstyle=\ttfamily\footnotesize,
  backgroundcolor=\color{{codegrey}},
  keywordstyle=\color{{blue!70!black}}\bfseries,
  commentstyle=\color{{green!40!black}}\itshape,
  stringstyle=\color{{red!70!black}},
  language=Python,
  showstringspaces=false,
  breaklines=true,
  numbers=left,
  numberstyle=\tiny\color{{gray}},
  frame=single,
  framesep=4pt,
  columns=fullflexible,
  tabsize=4,
}}
\title{{\texttt{{{safe_rel}}}}}
\date{{}}
\begin{{document}}
\maketitle
\lstinputlisting{{{src.as_posix()}}}
\end{{document}}
"""
    # Build in an isolated work dir so the auxiliary .aux/.log files don't
    # pollute .exports/.
    work = dst.with_suffix("")
    work_dir = work.parent / f"_tex_{work.name}"
    work_dir.mkdir(parents=True, exist_ok=True)
    tex_path = work_dir / f"{work.name}.tex"
    tex_path.write_text(tex_src, encoding="utf-8")

    try:
        proc = subprocess.run(
            [pdflatex, "-interaction=nonstopmode",
             "-output-directory", str(work_dir), str(tex_path)],
            capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace",  # Windows cp1252 chokes on pdflatex output
        )
        produced = work_dir / f"{work.name}.pdf"
        if produced.exists():
            shutil.move(str(produced), str(dst))
            shutil.rmtree(work_dir, ignore_errors=True)
            return True
        # Build failed — keep work_dir so user can inspect the .log.
        sys.stderr.write(
            f"  pdflatex failed for {src.name} (returncode={proc.returncode}); "
            f"see {work_dir}/{work.name}.log\n"
        )
        return False
    except Exception as e:
        sys.stderr.write(f"  pdflatex crash on {src.name}: {e}\n")
        return False


# ── Notebook <-> Markdown / HTML / PDF (via nbconvert) ───────────────────
_PANDOC_PRESENT: bool | None = None


def _have_pandoc() -> bool:
    global _PANDOC_PRESENT
    if _PANDOC_PRESENT is None:
        _PANDOC_PRESENT = shutil.which("pandoc") is not None
    return _PANDOC_PRESENT


def nb_export(src: Path, dst: Path, fmt: str) -> bool:
    """Invoke `jupyter nbconvert --to {fmt}` and move the output.

    nbconvert's `--to pdf` path goes notebook -> LaTeX -> PDF and uses
    pandoc to convert markdown cells to LaTeX. If pandoc isn't installed
    we skip PDF for notebooks (returning False) and the caller logs it as
    a skip rather than a failure. `--to markdown` and `--to html` work
    without pandoc.
    """
    # Import early so a missing-dep failure surfaces with a clear traceback
    # instead of an opaque nbconvert subprocess error later.
    import nbconvert
    _ = nbconvert.__version__

    if fmt == "pdf" and not _have_pandoc():
        sys.stderr.write(
            f"  nb->pdf skipped for {src.name}: pandoc not on PATH. "
            f"Install pandoc (https://pandoc.org/installing.html) and rerun "
            f"for notebook PDFs; .md/.html still produced.\n"
        )
        return False

    # nbconvert writes <name>.<ext> next to the source — capture and move.
    ext = {"markdown": "md", "html": "html", "pdf": "pdf"}[fmt]
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", fmt,
        "--output", dst.stem,
        "--output-dir", str(dst.parent),
        str(src),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                              encoding="utf-8", errors="replace")
    except Exception as e:
        sys.stderr.write(f"  nbconvert crash on {src.name}: {e}\n")
        return False

    produced = dst.parent / f"{dst.stem}.{ext}"
    if produced.exists():
        if produced != dst:
            shutil.move(str(produced), str(dst))
        return True
    sys.stderr.write(
        f"  nbconvert {fmt} failed for {src.name} "
        f"(rc={proc.returncode}):\n{proc.stderr[:500]}\n"
    )
    return False


# ── Driver ───────────────────────────────────────────────────────────────
def run(out_root: Path, formats: tuple[str, ...]) -> dict:
    """Convert every script and notebook into `out_root/{scripts,notebooks}/`."""
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "scripts").mkdir(parents=True)
    (out_root / "notebooks").mkdir(parents=True)

    stats = {"md": 0, "html": 0, "pdf": 0, "skipped_pdf": 0, "failed": 0}

    # --- Scripts ---
    for src in sorted(SCRIPTS.glob("*.py")):
        stem = src.stem
        if "md" in formats:
            py_to_md(src, out_root / "scripts" / f"{stem}.md")
            stats["md"] += 1
        if "html" in formats:
            py_to_html(src, out_root / "scripts" / f"{stem}.html")
            stats["html"] += 1
        if "pdf" in formats:
            ok = py_to_pdf(src, out_root / "scripts" / f"{stem}.pdf")
            stats["pdf" if ok else "skipped_pdf"] += 1

    # --- Notebooks ---
    for src in sorted(NOTEBOOKS.glob("*.ipynb")):
        stem = src.stem
        if "md" in formats:
            ok = nb_export(src, out_root / "notebooks" / f"{stem}.md", "markdown")
            if ok: stats["md"] += 1
            else:  stats["failed"] += 1
        if "html" in formats:
            ok = nb_export(src, out_root / "notebooks" / f"{stem}.html", "html")
            if ok: stats["html"] += 1
            else:  stats["failed"] += 1
        if "pdf" in formats:
            ok = nb_export(src, out_root / "notebooks" / f"{stem}.pdf", "pdf")
            stats["pdf" if ok else "skipped_pdf"] += 1
    return stats


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"output root (default {DEFAULT_OUT})")
    ap.add_argument("--only", nargs="+",
                    choices=["md", "html", "pdf"],
                    default=["md", "html", "pdf"],
                    help="formats to emit (default: all three)")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"Exporting to {args.out}/  formats={args.only}")
    stats = run(args.out, tuple(args.only))
    print(
        f"  md  : {stats['md']:3d}\n"
        f"  html: {stats['html']:3d}\n"
        f"  pdf : {stats['pdf']:3d}   "
        f"(skipped: {stats['skipped_pdf']}, failed: {stats['failed']})"
    )

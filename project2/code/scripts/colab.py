"""
colab.py — Colab/local setup with Drive-persistent outputs.

Pattern (mirrors the fys5429/PINN-BS notebooks):

    Cell 1 — Bootstrap (clone repo to ephemeral /content, chdir, add to path)
    Cell 2 — Paths     (mount Drive, route plots/results to Drive)

Public surface
--------------
    setup()                          # pip install + sys.path injection
    mount_drive()                    # one-time OAuth, returns Drive root
    out_dir(category, subdir=None)   # Drive-aware output path

`out_dir('plots', 'eda')` returns
    /content/drive/MyDrive/GITHUB-COLAB/fys5419/project2/code/plots/eda  (Colab)
    <repo>/project2/code/plots/eda                                       (local)

Notebooks should use `out_dir('plots', ...)` and `out_dir('results')` instead
of `Path("..")/'plots'/...`, so PDF outputs + sweep JSONs survive Colab
runtime restarts.
"""
from __future__ import annotations
import sys
import subprocess
from pathlib import Path


#: Drive layout: <DRIVE_ROOT>/<DRIVE_PROJECT>/<SUBPROJECT>/{plots, results, data}
DRIVE_ROOT      = "/content/drive/MyDrive/GITHUB-COLAB"
DRIVE_PROJECT   = "fys5419"
SUBPROJECT      = "project2/code"

REQUIREMENTS_DEFAULT = ("numpy", "pandas", "scipy", "matplotlib",
                        "yfinance", "pyarrow")


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _run(cmd: list[str], check: bool = True) -> int:
    print(">", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed (exit {proc.returncode}): {cmd!r}")
    return proc.returncode


# ── pip install + sys.path injection ─────────────────────────────────────
def setup(
    requirements: tuple[str, ...] = REQUIREMENTS_DEFAULT,
    install_deps: bool = True,
    verbose:      bool = True,
) -> Path:
    """Idempotent project setup. Returns the path of `<repo>/<subproject>`.

    Assumes the inline bootstrap cell in the notebook has already cloned
    the repo (on Colab) or that we are already inside the local repo.
    """
    if in_colab() and install_deps and requirements:
        _run([sys.executable, "-m", "pip", "install", "-q", *requirements],
             check=False)

    scripts_parent = _find_subproject_from_cwd(SUBPROJECT)
    sys.path.insert(0, str(scripts_parent))

    if verbose:
        env = "Colab" if in_colab() else "local"
        print(f"[colab.setup] env={env}")
        print(f"[colab.setup] cwd  = {Path.cwd()}")
        print(f"[colab.setup] path = {scripts_parent}  (added to sys.path)")

    return scripts_parent


# ── Drive mount + output paths ───────────────────────────────────────────
def mount_drive(verbose: bool = False) -> Path:
    """Mount Google Drive (no-op locally) and return the Drive project root.

    Drive project root = `<DRIVE_ROOT>/<DRIVE_PROJECT>/<SUBPROJECT>`,
    created if missing on first call.
    """
    if in_colab():
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")

    root = Path(DRIVE_ROOT) / DRIVE_PROJECT / SUBPROJECT
    if in_colab():
        root.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[colab.mount_drive] root = {root}")
    return root


def out_dir(category: str, subdir: str | None = None) -> Path:
    """Return the output directory for `category` (plots, results, data, ...).

    - On Colab: `<DRIVE_ROOT>/<DRIVE_PROJECT>/<SUBPROJECT>/<category>[/<subdir>]`
      (mounts Drive on first call). Outputs persist across runtime restarts.
    - Locally: `<repo>/<SUBPROJECT>/<category>[/<subdir>]`.

    The directory is created if it does not exist.
    """
    if in_colab():
        base = mount_drive(verbose=False) / category
    else:
        base = _find_subproject_from_cwd(SUBPROJECT) / category

    d = base if subdir is None else base / subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Local layout discovery ───────────────────────────────────────────────
def _find_subproject_from_cwd(subproject: str) -> Path:
    """Walk up from cwd looking for the folder that contains scripts/ + notebooks/."""
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "scripts").is_dir() and (p / "notebooks").is_dir():
            return p
    for p in (here, *here.parents):
        candidate = p / subproject
        if candidate.is_dir():
            return candidate
    raise RuntimeError(
        f"could not locate '{subproject}' from {here}; "
        f"call setup(subproject=...) with an explicit path"
    )

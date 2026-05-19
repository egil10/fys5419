"""
colab.py — One-line environment setup that works on Colab AND locally.

Put this at the top of every notebook:

    from scripts.colab import setup; setup()

…or, equivalently, the absolute minimum that also works before `scripts/`
is on `sys.path`:

    %pip -q install requests
    exec(__import__('urllib.request', fromlist=['x']).urlopen(
        'https://raw.githubusercontent.com/egil10/fys5419/main/'
        'project2/code/scripts/colab.py').read())
    setup()

Behaviour
---------
- On Colab:
    1. `git clone` (or `git -C ... pull`) the repo to `/content/<name>`
    2. `os.chdir` to `<repo>/project2/code/notebooks`
    3. add `<repo>/project2/code` to `sys.path` (so `from scripts.foo` works)
    4. `pip install -q` the project requirements (yfinance, pandas, ...)
- Locally: detect we are *not* on Colab and just ensure `sys.path` has the
  `code/` directory so notebook imports still work.

Idempotent — safe to call twice.
"""
from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path


REPO_URL_DEFAULT     = "https://github.com/egil10/fys5419.git"
REPO_NAME_DEFAULT    = "fys5419"
SUBPROJECT_DEFAULT   = "project2/code"     # `<repo>/<subproject>/{scripts,notebooks,...}`
REQUIREMENTS_DEFAULT = ("numpy", "pandas", "scipy", "matplotlib",
                        "yfinance", "pyarrow")


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _run(cmd: list[str], check: bool = True) -> int:
    """Run a shell command and stream its output."""
    print(">", " ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed (exit {proc.returncode}): {cmd!r}")
    return proc.returncode


def setup(
    repo_url:     str = REPO_URL_DEFAULT,
    repo_name:    str = REPO_NAME_DEFAULT,
    subproject:   str = SUBPROJECT_DEFAULT,
    requirements: tuple[str, ...] = REQUIREMENTS_DEFAULT,
    install_deps: bool = True,
    verbose:      bool = True,
) -> Path:
    """Bootstrap the notebook environment. Returns the path of `<repo>/<subproject>`.

    Locally: no clone, no pip; just ensures `<subproject>` is on `sys.path`.
    On Colab: clones or pulls, chdirs to `<subproject>/notebooks`, installs deps.
    """
    if in_colab():
        repo_dir = Path("/content") / repo_name
        if not repo_dir.exists():
            _run(["git", "clone", repo_url, str(repo_dir)])
        else:
            _run(["git", "-C", str(repo_dir), "pull"], check=False)

        notebooks_dir = repo_dir / subproject / "notebooks"
        os.chdir(notebooks_dir)

        if install_deps and requirements:
            _run([sys.executable, "-m", "pip", "install", "-q", *requirements],
                 check=False)

        scripts_parent = repo_dir / subproject
    else:
        # Local mode: walk up from the current working directory to find
        # the `<subproject>` folder that contains `scripts/` and `notebooks/`.
        scripts_parent = _find_local_subproject(subproject)

    sys.path.insert(0, str(scripts_parent))

    if verbose:
        env = "Colab" if in_colab() else "local"
        print(f"[colab.setup] env={env}")
        print(f"[colab.setup] cwd  = {Path.cwd()}")
        print(f"[colab.setup] path = {scripts_parent}  (added to sys.path)")

    return scripts_parent


def _find_local_subproject(subproject: str) -> Path:
    """Walk up from cwd looking for a folder ending with `subproject`."""
    here = Path.cwd().resolve()
    # First: are we already inside `<subproject>` (e.g. running a notebook
    # from `<subproject>/notebooks`)?
    for p in (here, *here.parents):
        if (p / "scripts").is_dir() and (p / "notebooks").is_dir():
            return p
    # Fallback: try `<root>/<subproject>` upward
    for p in (here, *here.parents):
        candidate = p / subproject
        if candidate.is_dir():
            return candidate
    raise RuntimeError(
        f"could not locate '{subproject}' from {here}; "
        f"call setup(subproject=...) with an explicit path"
    )

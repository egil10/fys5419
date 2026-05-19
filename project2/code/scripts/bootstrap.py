"""
bootstrap.py — All the Colab/local plumbing in one place.

Usage in any notebook (2 lines total):

    import os, urllib.request as _u
    exec((open('../scripts/bootstrap.py') if os.path.exists('../scripts/bootstrap.py') else _u.urlopen('https://raw.githubusercontent.com/egil10/fys5419/main/project2/code/scripts/bootstrap.py')).read())

On Colab: clones the repo into ephemeral `/content/fys5419` (or pulls if
present), `cd`s into `project2/code/notebooks`, adds `../` to `sys.path`,
and calls `scripts.colab.setup()` (pip install + summary print).

Locally: just adds `../` to `sys.path` and calls `scripts.colab.setup()`.

Idempotent. Safe to re-run.
"""
import os, sys

try:
    import google.colab  # noqa: F401
    _IN_COLAB = True
except ImportError:
    _IN_COLAB = False

if _IN_COLAB:
    _REPO = '/content/fys5419'
    if not os.path.exists(_REPO):
        os.system(f'git clone -q https://github.com/egil10/fys5419.git {_REPO}')
    else:
        os.system(f'git -C {_REPO} pull -q')
    os.chdir(f'{_REPO}/project2/code/notebooks')

if '..' not in sys.path:
    sys.path.insert(0, '..')

from scripts.colab import setup as _setup
_setup(verbose=True)
del _setup

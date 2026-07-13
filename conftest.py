"""
Pytest configuration for the notebook integration tests under ``docs/``.

The notebooks in this folder double as integration tests: ``make test-notebooks``
executes them with ``pytest --nbmake``. Each of them needs a large raster and/or
a cloud credential that is not always available, so a notebook is *skipped
automatically* unless everything it needs is present. This keeps the suite green
on a machine with no data and lets CI run exactly the subset of notebooks whose
inputs (data files in ``tests/data/`` and/or secrets in the environment) have
been provided.

How requirements are expressed
------------------------------
For every gated notebook we list one or more :class:`Requirement` groups. A
notebook runs only if **all** of its groups are satisfied, and a single group is
satisfied if **any** of these is true:

* one of its ``files`` exists under ``tests/data/`` (repo root), or
* one of its ``env`` environment variables is set (the notebook then downloads /
  authenticates by itself), or
* one of its ``paths`` credential files exists.

Notebooks that are **not** listed here always run.

Providing the inputs
--------------------
* ``tests/data/`` data files.
* PRISMA / EnMAP (UNEP IMEO Azure container): ``SAS_TOKEN``, ``AZURE_STORAGE_ACCOUNT``, ``CONTAINER_NAME``
* EMIT (NASA Earthdata):                      ``EARTHDATA_TOKEN``

These can be set directly as environment variables, or placed in a repo-root
``.env`` file (git-ignored) which this conftest loads automatically via
python-dotenv before running the notebooks. See ``.env.sample`` for the template.
In GitHub Actions they can be wired as repository secrets and exported as the
matching environment variables before ``make test-notebooks`` runs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# Resolve tests/data/ relative to this file (docs/conftest.py -> repo root -> tests/data/)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _REPO_ROOT / "tests" / "data"

# Load credentials/config from a repo-root .env file if present (and python-dotenv
# is installed) so they are available both for the gating below and for the
# notebook kernels (which inherit this process's environment). See .env.sample.
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass


def _file_available(path: Path) -> bool:
    """True if ``path`` exists as real data (not an un-smudged Git LFS pointer).

    Some example rasters may be stored with Git LFS. On a clone without git-lfs
    the file is present but is a tiny text pointer, which would make a notebook
    fail rather than skip. Treat such pointers as "not available" so the notebook
    is skipped cleanly.
    """
    if not path.exists():
        return False
    try:
        if path.stat().st_size < 1024:
            with open(path, "rb") as fh:
                if fh.read(40).startswith(b"version https://git-lfs"):
                    return False
    except OSError:
        return False
    return True


_ENMAP_TILE = "ENMAP01-____L1B-DT0000149931_20250820T075156Z_002_V010502_20250827T172144Z"


@dataclass
class Requirement:
    """A single requirement group (satisfied if *any* member is available)."""

    files: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    def satisfied(self) -> bool:
        if any(_file_available(_DATA_DIR / f) for f in self.files):
            return True
        if any(os.environ.get(e) for e in self.env):
            return True
        if any(Path(p).expanduser().exists() for p in self.paths):
            return True
        return False

    def describe(self) -> str:
        bits = []
        if self.files:
            bits.append(f"a data file in {_DATA_DIR} ({', '.join(self.files)})")
        if self.env:
            bits.append(f"one of env vars [{', '.join(self.env)}]")
        if self.paths:
            bits.append(f"one of credential files [{', '.join(self.paths)}]")
        return " OR ".join(bits)


# Keyed by notebook basename (basenames are unique across docs/).
NOTEBOOK_REQUIREMENTS: dict[str, list[Requirement]] = {
    # --- EMIT: local file or NASA Earthdata token ---------------------------
    "emit_example.ipynb": [
        Requirement(
            files=["EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc"],
            env=["EARTHDATA_TOKEN"],
        ),
    ],
    # --- EnMAP: local file or Azure download --------------------------------
    "enmap_example.ipynb": [
        Requirement(
            files=[f"EnMAP/{_ENMAP_TILE}/{_ENMAP_TILE}-METADATA.XML"],
            env=["SAS_TOKEN", "AZURE_STORAGE_ACCOUNT", "CONTAINER_NAME"],
        ),
    ],
    # --- PRISMA: local file or Azure download -------------------------------
    "prisma_example.ipynb": [
        Requirement(
            files=["PRISMA/PRS_L1_STD_OFFL_20250316071846_20250316071850_0001.he5"],
            env=["SAS_TOKEN", "AZURE_STORAGE_ACCOUNT", "CONTAINER_NAME"],
        ),
    ],
}


# Notebooks that are *always* skipped, regardless of available data or
# credentials, because they cannot run for reasons unrelated to missing local
# inputs (e.g. they depend on a decommissioned service). Keyed by basename.
ALWAYS_SKIP: dict[str, str] = {}


def pytest_collection_modifyitems(config, items):
    """Skip notebook tests whose required data files / credentials are missing."""
    for item in items:
        if item.fspath.ext != ".ipynb":
            continue

        name = Path(item.fspath).name

        always_skip_reason = ALWAYS_SKIP.get(name)
        if always_skip_reason:
            item.add_marker(pytest.mark.skip(reason=always_skip_reason))
            continue

        requirements = NOTEBOOK_REQUIREMENTS.get(name)
        if not requirements:
            continue

        for req in requirements:
            if not req.satisfied():
                item.add_marker(
                    pytest.mark.skip(reason=f"missing notebook input: needs {req.describe()}")
                )
                break

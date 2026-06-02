"""
Conftest for notebook integration tests (docs/ directory).

Each notebook is skipped automatically unless the data it needs is available:
either the required files already exist under ``tests/data/``, OR the
credentials needed to download them are present in the environment (in which
case the notebook downloads the data itself when it runs).

Credentials (set as environment variables locally, or as GitHub Actions
secrets exported into the job environment):

* PRISMA / EnMAP (UNEP IMEO Azure container): ``SAS_TOKEN``
  (plus ``AZURE_STORAGE_ACCOUNT`` and ``CONTAINER_NAME`` for the actual
  download).
* EMIT (NASA Earthdata): ``EARTHDATA_TOKEN``.

These can be set directly as environment variables, or placed in a repo-root
``.env`` file (git-ignored) which this conftest loads automatically via
python-dotenv before the notebooks run. See ``.env.sample`` for the template.
"""

import os
from pathlib import Path

import pytest

# Resolve paths relative to this file (docs/conftest.py -> project root).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "tests" / "data"

# Load credentials/config from a repo-root .env file if present (and
# python-dotenv is installed) so they are available both for the gating below
# and for the notebook kernels (which inherit this process's environment).
# See .env.sample.
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

_ENMAP_TILE = "ENMAP01-____L1B-DT0000149931_20250820T075156Z_002_V010502_20250827T172144Z"

# Map notebook basenames to the data files they require (relative to tests/data/).
NOTEBOOK_REQUIRED_FILES: dict[str, list[str]] = {
    "emit_example.ipynb": [
        "EMIT_L1B_RAD_001_20220827T060753_2223904_013.nc",
        "EMIT_L1B_OBS_001_20220827T060753_2223904_013.nc",
    ],
    "enmap_example.ipynb": [
        f"EnMAP/{_ENMAP_TILE}/{_ENMAP_TILE}-METADATA.XML",
        f"EnMAP/{_ENMAP_TILE}/{_ENMAP_TILE}-SPECTRAL_IMAGE_VNIR.TIF",
        f"EnMAP/{_ENMAP_TILE}/{_ENMAP_TILE}-SPECTRAL_IMAGE_SWIR.TIF",
    ],
    "prisma_example.ipynb": [
        "PRISMA/PRS_L1_STD_OFFL_20250316071846_20250316071850_0001.he5",
    ],
}

# Map notebook basenames to the environment variable that, when set, lets the
# notebook download its own data -- so the test runs even if the files are not
# present locally (e.g. on a fresh CI checkout with secrets configured).
NOTEBOOK_DOWNLOAD_ENV: dict[str, str] = {
    "emit_example.ipynb": "EARTHDATA_TOKEN",
    "enmap_example.ipynb": "SAS_TOKEN",
    "prisma_example.ipynb": "SAS_TOKEN",
}


def pytest_collection_modifyitems(config, items):
    """Skip notebook tests whose data is neither present nor downloadable."""
    for item in items:
        if item.fspath.ext != ".ipynb":
            continue

        notebook_name = Path(item.fspath).name
        required_files = NOTEBOOK_REQUIRED_FILES.get(notebook_name)
        if required_files is None:
            continue

        # If the credentials to download the data are set, let the notebook run
        # (it downloads the missing files itself).
        download_env = NOTEBOOK_DOWNLOAD_ENV.get(notebook_name)
        if download_env and os.environ.get(download_env):
            continue

        for rel_path in required_files:
            full_path = _DATA_DIR / rel_path
            if not full_path.exists():
                reason = f"Required data file not found: {full_path}"
                if download_env:
                    reason += f" (set ${download_env} to download it instead)"
                item.add_marker(pytest.mark.skip(reason=reason))
                break

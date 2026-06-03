"""
Pytest configuration for marshsi tests.

This file contains pytest hooks and fixtures shared across all test modules.
"""

import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

# ── Real EMIT scene fixture (fetched from the georeader repo) ────────────
#
# The 200x200 EMIT scene fixtures live in the georeader repo under
# tests/data/ (committed as plain, non-LFS blobs). We fetch the RAD scene plus
# its OBS and L2A_MASK companions from GitHub raw so EMITImage can auto-discover
# the companions by filename. Files are cached in the system temp dir, so they
# are downloaded at most once per machine. Tests that depend on these are
# skipped when GitHub is unreachable (e.g. offline CI).

_GEOREADER_RAW_BASE = (
    "https://raw.githubusercontent.com/spaceml-org/georeader/main/tests/data"
)
_EMIT_FIXTURE_FILES = {
    "rad": "EMIT_L1B_RAD_001_20220827T060753_9999999_999.nc",
    "obs": "EMIT_L1B_OBS_001_20220827T060753_9999999_999.nc",
    "mask": "EMIT_L2A_MASK_001_20220827T060753_9999999_999.nc",
}
_EMIT_FIXTURE_CACHE = Path(tempfile.gettempdir()) / "marshsi_emit_fixtures"


def _download_emit_fixture(filename: str) -> Path:
    """Download one georeader EMIT fixture into the local cache (idempotent)."""
    _EMIT_FIXTURE_CACHE.mkdir(parents=True, exist_ok=True)
    dst = _EMIT_FIXTURE_CACHE / filename
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    url = f"{_GEOREADER_RAW_BASE}/{filename}"
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, tmp)  # noqa: S310 (trusted github URL)
    except (urllib.error.URLError, OSError) as exc:
        if tmp.exists():
            tmp.unlink()
        pytest.skip(f"could not fetch EMIT fixture {filename} from github: {exc}")
    os.replace(tmp, dst)
    return dst


@pytest.fixture(scope="session")
def emit_rad_fixture_path() -> Path:
    """Path to the cached EMIT L1B RAD scene, with OBS + L2A_MASK companions
    fetched alongside it so EMITImage's filename-based discovery works.

    Returns the RAD ``.nc`` path. Skips the test if GitHub is unreachable.
    """
    paths = {key: _download_emit_fixture(name) for key, name in _EMIT_FIXTURE_FILES.items()}
    return paths["rad"]


@pytest.fixture
def emit_image(emit_rad_fixture_path):
    """A fresh georeader EMITImage backed by the real RAD fixture."""
    from georeader.readers import emit

    return emit.EMITImage(str(emit_rad_fixture_path))


# ── Shared infrastructure fixtures ──────────────────────────────────────


@pytest.fixture
def mock_logger():
    """Mock logger that supports standard logging interface."""
    logger = MagicMock()
    return logger


# ── Shared geometry fixtures ────────────────────────────────────────────


@pytest.fixture
def sample_polygon():
    """A 1x1 degree polygon centered on (-121.5, 37.5)."""
    return Polygon([(-122, 37), (-122, 38), (-121, 38), (-121, 37)])


@pytest.fixture
def sample_multipolygon(sample_polygon):
    return MultiPolygon([sample_polygon])


@pytest.fixture
def sample_point():
    return Point(-121.5, 37.5)

"""
Pytest configuration for marshsi tests.

This file contains pytest hooks and fixtures shared across all test modules.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

# ── Real EMIT plume fixture (committed via Git LFS) ──────────────────────
#
# A dense 350x350 native-resolution EMIT subset centered on a real methane plume,
# built by notebooks/emit_fixture.ipynb. The three EMIT products (RAD/OBS/MASK)
# and the 7 reference matched-filter products (UTM COGs) live under
# tests/data/emit_plume_fixture/ and are tracked with Git LFS. Tests that use them
# are skipped (with a helpful message) when the files are absent or are still
# unmaterialized LFS pointers — run `git lfs pull` (or `make test`) to fetch them.

_FIXTURE_DIR = Path(__file__).parent / "data" / "emit_plume_fixture"
_FIXTURE_RAD = _FIXTURE_DIR / "EMIT_L1B_RAD_001_20220827T060753_9999999_001.nc"
_PRODUCTS_DIR = _FIXTURE_DIR / "products"

# The 7 products written by emit_fixture.ipynb (UTM GeoTensors saved as COGs).
EMIT_PRODUCT_NAMES = (
    "mf_classic",
    "mf_extended",
    "mf_combo",
    "mf_extended_filtered",
    "rad",
    "mag1c",
    "albedo",
)


def _require_emit_fixture() -> None:
    """Skip the calling test if the LFS-tracked EMIT fixture is not materialized."""
    if not _FIXTURE_RAD.exists():
        pytest.skip(
            "EMIT plume fixture missing — run `git lfs pull` (or `make test`) to fetch "
            f"{_FIXTURE_RAD.relative_to(Path(__file__).parent.parent)}."
        )
    # A real RAD fixture is ~129 MB; an unfetched LFS pointer is a ~130-byte text file.
    if _FIXTURE_RAD.stat().st_size < 10_000:
        pytest.skip(
            "EMIT plume fixture is an unmaterialized Git LFS pointer — run `git lfs pull`."
        )


@pytest.fixture(scope="module")
def emit_image():
    """A georeader EMITImage backed by the committed plume fixture."""
    _require_emit_fixture()
    from georeader.readers import emit

    return emit.EMITImage(str(_FIXTURE_RAD))


@pytest.fixture(scope="module")
def emit_plume_polygon():
    """The largest plume polygon (WGS84) for the fixture scene — the window is
    centered on it. Used by compute_emit and to overlay/locate the plume."""
    import geopandas as gpd

    gpkg = Path(__file__).parent / "data" / "plume_vetting" / "plumes_emit.gpkg"
    if not gpkg.exists():
        pytest.skip(f"plume polygons missing: {gpkg}")
    plumes = gpd.read_file(gpkg).to_crs("EPSG:4326")
    # Rank by area in a projected CRS (geographic-CRS area is invalid/warns).
    areas = plumes.to_crs(plumes.estimate_utm_crs()).geometry.area
    return plumes.geometry.iloc[int(areas.idxmax())]


@pytest.fixture(scope="module")
def emit_utm_products(emit_image):
    """Recompute the 7 matched-filter products in UTM exactly as
    notebooks/emit_fixture.ipynb does: retrieve in the raw sensor frame
    (georeferenced=False), reproject to UTM and georeference each product with
    fill_value_default=-1. Module-scoped so the retrievals run once.
    """
    import georeader

    from marshsi.emit.mag1c_emit import mag1c_emit
    from marshsi.emit.retrieval_upv_emit import AT_MF_total_EMIT

    crs_utm = georeader.get_utm_epsg(emit_image.footprint("EPSG:4326"))
    eutm = emit_image.to_crs(crs_utm)
    mf_classic, mf_extended, mf_combo, mf_extended_filtered, rad = AT_MF_total_EMIT(
        emit_image, georeferenced=False
    )
    mfoutput, albedo = mag1c_emit(emit_image, georeferenced=False, display_pbar=False)
    g = lambda a: eutm.georreference(a, fill_value_default=-1)
    return {
        "mf_classic": g(mf_classic),
        "mf_extended": g(mf_extended),
        "mf_combo": g(mf_combo),
        "mf_extended_filtered": g(mf_extended_filtered),
        "rad": g(rad),
        "mag1c": g(mfoutput),
        "albedo": g(albedo),
    }


@pytest.fixture(scope="module")
def emit_products_dir():
    """Directory holding the committed reference product COGs."""
    _require_emit_fixture()
    return _PRODUCTS_DIR


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

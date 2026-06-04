"""
Golden-product regression tests for the EMIT matched filters.

The committed reference COGs in ``tests/data/emit_plume_fixture/products/`` were
produced by ``notebooks/emit_fixture.ipynb`` (UTM GeoTensors). Here we recompute
each product from the fixture with the exact same pipeline (``emit_utm_products``
in conftest) and assert it matches the reference, so any change to the matched
filters that perturbs the outputs is caught.

The extended WMF (``mf_extended``) is the operationally critical product and gets
an extra check that it actually carries plume signal.
"""

import numpy as np
import pytest
from georeader.geotensor import GeoTensor

PRODUCT_NAMES = [
    "mf_classic",
    "mf_extended",
    "mf_combo",
    "mf_extended_filtered",
    "rad",
    "mag1c",
    "albedo",
]


def _valid_pair(recomputed: GeoTensor, golden: GeoTensor):
    """Aligned valid (non-fill, finite) pixel values from both GeoTensors."""
    a = np.asarray(recomputed.values)
    b = np.squeeze(np.asarray(golden.values))  # load_file gives (1, H, W)
    m = (
        (a != recomputed.fill_value_default)
        & (b != golden.fill_value_default)
        & np.isfinite(a)
        & np.isfinite(b)
    )
    return a[m], b[m]


@pytest.mark.parametrize("name", PRODUCT_NAMES)
def test_product_matches_reference(name, emit_utm_products, emit_products_dir):
    recomputed = emit_utm_products[name]
    golden = GeoTensor.load_file(str(emit_products_dir / f"{name}.tif"))

    # Georeferencing (crs, transform, spatial shape) must match exactly.
    assert recomputed.same_extent(golden), f"{name}: georeferencing differs from reference"

    a, b = _valid_pair(recomputed, golden)
    assert a.size > 0, f"{name}: no valid pixels to compare"
    maxdiff = float(np.abs(a - b).max())
    assert np.allclose(a, b, rtol=1e-3, atol=1e-3), f"{name}: max abs diff {maxdiff:.4g}"


def test_extended_wmf_carries_plume_signal(emit_utm_products):
    """The extended WMF (operational product) must show a clear positive
    enhancement on this plume scene — not just noise around zero."""
    gt = emit_utm_products["mf_extended"]
    v = gt.values
    m = (v != gt.fill_value_default) & np.isfinite(v)
    assert m.any()
    # WMF is in ppm; the plume produces a clear positive tail well above the
    # near-zero background. (Fixture p98 ~ 0.045 ppm.)
    assert np.percentile(v[m], 99) > 0.03


def test_reference_products_are_utm(emit_utm_products):
    """Reference products are stored/compared in a projected (UTM) CRS."""
    crs = emit_utm_products["mf_extended"].crs
    from rasterio.crs import CRS as _CRS

    assert _CRS.from_user_input(crs).is_projected

"""
End-to-end test for marshsi.plume_vetting.compute_emit on the real plume fixture.

Chains the whole EMIT path: fixture EMITImage + a CMF (the mag1c product, ppm·m)
+ the real plume polygon -> compute_emit -> per-polygon vetting scores.
"""

import numpy as np
from shapely.geometry import Polygon

from marshsi import plume_vetting

# Keys every scored polygon carries (see plume_vetting.compute, scores_per_polygon).
EXPECTED_KEYS = {
    "D_norm",
    "alpha_con_len",
    "top_pairs",
    "coef",
    "fit_sig",
    "ratio",
    "wl_only_in_fit",
}


def test_compute_emit_scores_in_footprint_plume(
    emit_image, emit_utm_products, emit_plume_polygon
):
    cmf = emit_utm_products["mag1c"]  # matched filter in ppm·m

    results = plume_vetting.compute_emit(
        emit_image, cmf, [emit_plume_polygon], num_pts=10, random_seed=0
    )

    assert isinstance(results, dict)
    assert 0 in results, "the in-footprint plume polygon was not scored"
    r = results[0]
    assert EXPECTED_KEYS.issubset(r.keys())
    assert np.isfinite(r["D_norm"])
    assert np.isfinite(r["alpha_con_len"])


def test_compute_emit_skips_off_footprint_polygon(emit_image, emit_utm_products):
    cmf = emit_utm_products["mag1c"]
    # A tiny polygon far from the scene (near 0,0) has zero in-plume pixels.
    far = Polygon([(0.0, 0.0), (0.0, 0.01), (0.01, 0.01), (0.01, 0.0)])

    results = plume_vetting.compute_emit(
        emit_image, cmf, [far], num_pts=10, random_seed=0
    )

    assert 0 not in results  # 0 in-plume pixels -> polygon skipped

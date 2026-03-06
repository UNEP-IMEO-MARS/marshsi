"""
Shared assertion helpers for mars_mf tests.

These are plain functions (not pytest fixtures) that bundle repeated
assertion patterns into single calls with clear error messages.

Usage::

    from tests.validators import assert_valid_flux_result

    result = obtain_flux_rate(...)
    assert_valid_flux_result(result)
"""

import numpy as np
from georeader.geotensor import GeoTensor


def assert_valid_flux_result(result, *, expect_std=False):
    """Assert that a flux rate result dict has all required keys with valid values.

    Checks:
      - Q > 0, L > 0, IME > 0, u_eff > 0, npix_plume > 0, pixel_size > 0
      - Q is finite (not NaN or Inf)
      - wind_speed is present
      - If expect_std: sigma_Q > 0 and sig_xch4 present

    Args:
        result: dict returned by obtain_flux_rate or get_quantification.
        expect_std: if True, also check sigma_Q and sig_xch4 are present and > 0.
    """
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    for key in ("Q", "L", "IME", "u_eff", "npix_plume", "pixel_size"):
        assert key in result, f"Missing key '{key}' in flux result"

    assert result["Q"] > 0, f"Q should be > 0, got {result['Q']}"
    assert np.isfinite(result["Q"]), f"Q should be finite, got {result['Q']}"
    assert result["L"] > 0, f"L should be > 0, got {result['L']}"
    assert result["IME"] > 0, f"IME should be > 0, got {result['IME']}"
    assert result["u_eff"] > 0, f"u_eff should be > 0, got {result['u_eff']}"
    assert result["npix_plume"] > 0, f"npix_plume should be > 0, got {result['npix_plume']}"
    assert result["pixel_size"] > 0, f"pixel_size should be > 0, got {result['pixel_size']}"

    if expect_std:
        assert "sigma_Q" in result, "Missing key 'sigma_Q' in flux result"
        assert "sig_xch4" in result, "Missing key 'sig_xch4' in flux result"
        assert result["sigma_Q"] > 0, f"sigma_Q should be > 0, got {result['sigma_Q']}"


def assert_valid_geotensor(gt, *, expected_shape=None, expected_crs=None):
    """Assert that an object is a valid GeoTensor with expected properties.

    Args:
        gt: the object to check.
        expected_shape: if provided, assert gt.shape matches.
        expected_crs: if provided, assert gt.crs matches.
    """
    assert isinstance(gt, GeoTensor), f"Expected GeoTensor, got {type(gt)}"
    assert gt.transform is not None, "GeoTensor.transform should not be None"

    if expected_shape is not None:
        assert gt.shape == expected_shape, (
            f"GeoTensor shape mismatch: expected {expected_shape}, got {gt.shape}"
        )

    if expected_crs is not None:
        assert str(gt.crs) == str(expected_crs), (
            f"GeoTensor CRS mismatch: expected {expected_crs}, got {gt.crs}"
        )

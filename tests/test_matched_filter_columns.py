"""Tests for marshsi.matched_filters_upv — columns the per-column matched filter cannot fit."""
import numpy as np
import pytest

from marshsi.matched_filters_upv import AT_MF_select_window_alt, compute_mf_standard


def _synthetic_image(h: int, w: int, b: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Radiance-like image (H, W, B) with a random band covariance, and a target spectrum (B,)."""
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(b, b)) / np.sqrt(b)
    mean = rng.uniform(5, 10, size=b)
    img = mean + rng.normal(size=(h, w, b)) @ mixing.T
    target = -rng.uniform(0, 0.05, size=b)
    return img, target


def test_mf_skips_empty_column(caplog):
    img, target = _synthetic_image(40, 5, 20)
    img[:, 1] = np.nan  # all-invalid column
    img[1:, 3] = np.nan  # single valid row

    expected = AT_MF_select_window_alt(np.delete(img, [1, 3], axis=1), target)

    with caplog.at_level("WARNING", logger="marshsi.matched_filters_upv"):
        mf = AT_MF_select_window_alt(img, target)

    assert np.isnan(mf[:, [1, 3]]).all()
    np.testing.assert_array_equal(mf[:, [0, 2, 4]], expected)
    assert "skipped 2/5 columns" in caplog.text


def test_mf_two_valid_rows_unchanged():
    img, target = _synthetic_image(40, 3, 20)
    img[2:, 1] = np.nan  # exactly two valid rows

    mf = AT_MF_select_window_alt(img, target)

    np.testing.assert_array_equal(mf[:2, 1], compute_mf_standard(img[:2, 1], target))
    assert np.isnan(mf[2:, 1]).all()


@pytest.mark.parametrize("target_ndim", [1, 2])
def test_mf_matches_compute_mf_standard_per_column(target_ndim):
    img, target = _synthetic_image(30, 4, 10)
    target_arg = np.tile(target, (4, 1)) if target_ndim == 2 else target

    mf = AT_MF_select_window_alt(img, target_arg)

    for i in range(4):
        np.testing.assert_array_equal(mf[:, i], compute_mf_standard(img[:, i], target))

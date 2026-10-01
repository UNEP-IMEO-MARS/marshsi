"""Tests for marshsi.matched_filters_upv — matched filter properties."""
import numpy as np

from marshsi.matched_filters_upv import compute_mf_standard


def _synthetic_image(h: int, w: int, b: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Radiance-like image (H, W, B) with a random band covariance, and a target spectrum (B,)."""
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(b, b)) / np.sqrt(b)
    mean = rng.uniform(5, 10, size=b)
    img = mean + rng.normal(size=(h, w, b)) @ mixing.T
    target = -rng.uniform(0, 0.05, size=b)
    return img, target


def test_mf_gain_invariant():
    # t = mu * k scales with the band gains, so the matched filter cancels any per-band gain.
    col, target = _synthetic_image(500, 1, 60)
    col = col[:, 0]
    gains = np.random.default_rng(1).uniform(0.9, 1.1, size=col.shape[1])

    mf = compute_mf_standard(col, target)
    mf_gain = compute_mf_standard(col * gains, target)

    np.testing.assert_allclose(mf_gain, mf, rtol=1e-9, atol=1e-12)

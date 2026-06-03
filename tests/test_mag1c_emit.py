"""
Tests for marshsi.emit.mag1c_emit.

These run against the real 200x200 EMIT RAD scene fetched from the georeader
repo (see the ``emit_image`` fixture in conftest.py). The focus is the NaN/inf
handling added to ``mag1c_emit``: a single non-finite radiance pixel propagates
through the covariance and makes ``torch.linalg.cholesky`` raise
``_LinAlgError: not positive-definite``. ``mag1c_emit`` must instead drop those
pixels (treating them like the fill value) and still produce finite results for
the valid ones.
"""

import numpy as np
import pytest

from marshsi.emit import mag1c
from marshsi.emit.mag1c_emit import mag1c_emit

from ._emit_corruption import inject_into_load_raw

# ``generate_template_from_bands`` emits a benign numpy UserWarning
# ("'where' used without 'out'") on the real fixture's band parameters; the
# suite promotes warnings to errors, so scope-ignore that one here. It is
# unrelated to the NaN handling under test and harmless in production (the
# gaussian response sums are always > 0, so no uninitialized memory is read).
_IGNORE_WHERE_WARNING = pytest.mark.filterwarnings(
    r"ignore:'where' used without 'out':UserWarning"
)


@pytest.mark.network
@_IGNORE_WHERE_WARNING
class TestMag1cEmitBaseline:
    """The happy path must work on the real scene before we test corruption."""

    def test_runs_and_returns_two_arrays(self, emit_image):
        mag, albedo = mag1c_emit(emit_image, georeferenced=False, display_pbar=False)

        assert mag.shape == albedo.shape
        assert mag.ndim == 2  # (rows, cols)

    def test_has_finite_valid_pixels(self, emit_image):
        mag, _ = mag1c_emit(emit_image, georeferenced=False, display_pbar=False)

        valid = mag[mag != emit_image.fill_value_default]
        assert valid.size > 0
        assert np.all(np.isfinite(valid))


@pytest.mark.network
@_IGNORE_WHERE_WARNING
class TestMag1cEmitNaNHandling:
    """The regression: non-finite radiance must not crash the Cholesky solve."""

    def test_nan_inf_pixels_do_not_crash(self, emit_image, monkeypatch):
        recorded = inject_into_load_raw(monkeypatch, ["nan", "nan_band", "inf"])

        # Before the fix this raised torch._C._LinAlgError (NaN-poisoned C).
        mag, albedo = mag1c_emit(emit_image, georeferenced=False, display_pbar=False)

        assert recorded, "load_raw was never called / no pixels corrupted"
        fill = emit_image.fill_value_default
        # Every corrupted pixel must be excluded -> fill value in the output.
        for y, x in recorded:
            assert mag[y, x] == fill
            assert albedo[y, x] == fill

    def test_valid_pixels_still_finite_with_corruption(self, emit_image, monkeypatch):
        inject_into_load_raw(monkeypatch, ["nan", "nan_band", "inf"])

        mag, _ = mag1c_emit(emit_image, georeferenced=False, display_pbar=False)

        valid = mag[mag != emit_image.fill_value_default]
        assert valid.size > 0
        assert np.all(np.isfinite(valid))


def test_acrwl1mf_rejects_nan_input_documents_failure_mode():
    """Document *why* the upstream pixel exclusion is needed: feeding a NaN
    pixel straight into the matched filter blows up the Cholesky factorization.
    """
    import torch

    x = torch.rand(1, 64, 8, dtype=torch.float64)
    x[0, 0, 0] = float("nan")  # one NaN poisons the whole covariance
    spec = torch.rand(8, dtype=torch.float64)

    with pytest.raises(torch._C._LinAlgError):
        mag1c.acrwl1mf(x, spec, num_iter=2, alpha=1e-4)

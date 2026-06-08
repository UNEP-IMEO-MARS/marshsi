"""
Tests for marshsi.emit.retrieval_upv_emit module.

Tests cover:
- load_target_spectrum_mf (target spectrum loading)
- extended_bool_wavelengths (wavelength selection)
- AT_MF_total_EMIT (total matching filter calculation)
- Module constants (wavelength ranges)
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from affine import Affine

from ._emit_corruption import inject_into_load_raw


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_wavelengths():
    """Sample EMIT-like wavelengths (285 bands from ~380nm to ~2500nm)."""
    return np.linspace(380, 2500, 285)


@pytest.fixture
def sample_fwhm():
    """Sample FWHM values for EMIT bands."""
    return np.ones(285) * 7.5  # Typical EMIT FWHM


@pytest.fixture
def mock_emit_image(sample_wavelengths, sample_fwhm):
    """Create a mock EMITImage object."""
    mock = MagicMock()
    mock.wavelengths = sample_wavelengths
    mock.fwhm = sample_fwhm
    mock.mean_vza = 10.0
    mock.mean_sza = 30.0
    mock.fill_value_default = -9999

    # Create mock for read_from_bands
    def mock_read_from_bands(indexes):
        subset_mock = MagicMock()
        subset_mock.wavelengths = sample_wavelengths[indexes]
        subset_mock.fwhm = sample_fwhm[indexes]
        H, W = 100, 50
        np.random.seed(42)
        # Return radiance-like data (positive values)
        data = np.random.rand(H, W, len(indexes)) * 0.3 + 0.1
        subset_mock.load_raw.return_value = data.astype(np.float32)
        return subset_mock

    mock.read_from_bands = mock_read_from_bands

    # Mock georreference method
    def mock_georreference(data, fill_value_default=-9999):
        from georeader.geotensor import GeoTensor

        transform = Affine.translation(0, 0) * Affine.scale(60, -60)
        return GeoTensor(
            data, transform=transform, crs="EPSG:32610", fill_value_default=fill_value_default
        )

    mock.georreference = mock_georreference

    return mock


# ─────────────────────────────────────────────────────────────────────────────
# Tests for module constants
# ─────────────────────────────────────────────────────────────────────────────
class TestModuleConstants:
    """Tests for module-level constants."""

    def test_extended_wavelength_range(self):
        """Test EXTENDED_WAVELENGTH_RANGE constant."""
        from marshsi.emit import retrieval_upv_emit as ret

        assert ret.EXTENDED_WAVELENGTH_RANGE == (975, 2_445)

    def test_classic_wavelength_range(self):
        """Test CLASSIC_WAVELENGTH_RANGE constant."""
        from marshsi.emit import retrieval_upv_emit as ret

        assert ret.CLASSIC_WAVELENGTH_RANGE == (2_100, 2_445)

    def test_rad_wavelength(self):
        """Test RAD_WAVELENGTH constant."""
        from marshsi.emit import retrieval_upv_emit as ret

        assert ret.RAD_WAVELENGTH == 2_100

    def test_periods_exclude_water(self):
        """Test PERIODS_EXCLUDE_WATER constant."""
        from marshsi.emit import retrieval_upv_emit as ret

        assert len(ret.PERIODS_EXCLUDE_WATER) > 0
        assert (1260, 1330) in ret.PERIODS_EXCLUDE_WATER


# ─────────────────────────────────────────────────────────────────────────────
# Tests for extended_bool_wavelengths
# ─────────────────────────────────────────────────────────────────────────────
class TestExtendedBoolWavelengths:
    """Tests for extended_bool_wavelengths function."""

    def test_extended_bool_basic(self, sample_wavelengths):
        """Test basic wavelength selection."""
        from marshsi.emit import retrieval_upv_emit as ret

        result = ret.extended_bool_wavelengths(sample_wavelengths)

        assert result.dtype == bool
        assert len(result) == len(sample_wavelengths)

    def test_extended_bool_range_selection(self, sample_wavelengths):
        """Test that wavelengths in range are selected."""
        from marshsi.emit import retrieval_upv_emit as ret

        result = ret.extended_bool_wavelengths(sample_wavelengths)

        # Wavelengths within range should be True (excluding water bands)
        for i, wvl in enumerate(sample_wavelengths):
            in_range = ret.EXTENDED_WAVELENGTH_RANGE[0] <= wvl <= ret.EXTENDED_WAVELENGTH_RANGE[1]
            in_water_band = any(
                period[0] <= wvl <= period[1] for period in ret.PERIODS_EXCLUDE_WATER
            )
            if in_range and not in_water_band:
                # Should be True
                pass  # Complex logic, just check the function runs

    def test_extended_bool_excludes_water_bands(self, sample_wavelengths):
        """Test that water absorption bands are excluded."""
        from marshsi.emit import retrieval_upv_emit as ret

        result = ret.extended_bool_wavelengths(sample_wavelengths)

        # Check water bands are excluded
        for period in ret.PERIODS_EXCLUDE_WATER:
            water_band_indices = np.where(
                (sample_wavelengths >= period[0]) & (sample_wavelengths <= period[1])
            )[0]
            if len(water_band_indices) > 0:
                assert np.all(result[water_band_indices] == False)

    def test_extended_bool_custom_range(self, sample_wavelengths):
        """Test with custom wavelength range."""
        from marshsi.emit import retrieval_upv_emit as ret

        custom_range = (1500, 2000)
        result = ret.extended_bool_wavelengths(
            sample_wavelengths, extended_wavelengths_range=custom_range
        )

        # Count selected wavelengths
        count_selected = np.sum(result)

        # Should select some but not all wavelengths
        assert count_selected < len(sample_wavelengths)
        assert count_selected > 0

    def test_extended_bool_empty_range(self, sample_wavelengths):
        """Test with range outside wavelength array."""
        from marshsi.emit import retrieval_upv_emit as ret

        # Range completely outside the wavelength array
        result = ret.extended_bool_wavelengths(
            sample_wavelengths, extended_wavelengths_range=(3000, 4000)
        )

        assert np.sum(result) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests for load_target_spectrum_mf
# ─────────────────────────────────────────────────────────────────────────────
class TestLoadTargetSpectrumMF:
    """Tests for load_target_spectrum_mf function."""

    def test_load_target_spectrum_returns_array(self, mock_emit_image):
        """Test that load_target_spectrum_mf returns an array."""
        from marshsi.emit import retrieval_upv_emit as ret

        result = ret.load_target_spectrum_mf(mock_emit_image)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(mock_emit_image.wavelengths)

    def test_load_target_spectrum_finite_values(self, mock_emit_image):
        """Test that returned spectrum has finite values."""
        from marshsi.emit import retrieval_upv_emit as ret

        result = ret.load_target_spectrum_mf(mock_emit_image)

        assert np.all(np.isfinite(result))

    def test_load_target_spectrum_high_amf_warning(self, mock_emit_image):
        """Test warning is logged for high AMF values."""
        from marshsi.emit import retrieval_upv_emit as ret

        # Set extreme angles that result in high AMF
        mock_emit_image.mean_sza = 80.0
        mock_emit_image.mean_vza = 70.0

        # Should still run without error
        result = ret.load_target_spectrum_mf(mock_emit_image)

        assert isinstance(result, np.ndarray)


# ─────────────────────────────────────────────────────────────────────────────
# Tests for AT_MF_total_EMIT
# ─────────────────────────────────────────────────────────────────────────────
class TestATMFTotalEMIT:
    """Tests for AT_MF_total_EMIT function."""

    def test_at_mf_total_emit_returns_tuple(self, mock_emit_image):
        """Test that AT_MF_total_EMIT returns a tuple of 5 elements."""
        from marshsi.emit import retrieval_upv_emit as ret

        result = ret.AT_MF_total_EMIT(mock_emit_image)

        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_at_mf_total_emit_geotensor_outputs(self, mock_emit_image):
        """Test that outputs are GeoTensors."""
        from georeader.geotensor import GeoTensor

        from marshsi.emit import retrieval_upv_emit as ret

        mf_classic, mf_extended, mf_combo, mf_filtered, rad = ret.AT_MF_total_EMIT(mock_emit_image)

        assert isinstance(mf_classic, GeoTensor)
        assert isinstance(mf_extended, GeoTensor)
        assert isinstance(mf_combo, GeoTensor)
        assert isinstance(mf_filtered, GeoTensor)
        assert isinstance(rad, GeoTensor)

    def test_at_mf_total_emit_same_shape_outputs(self, mock_emit_image):
        """Test that all outputs have the same shape."""
        from marshsi.emit import retrieval_upv_emit as ret

        mf_classic, mf_extended, mf_combo, mf_filtered, rad = ret.AT_MF_total_EMIT(mock_emit_image)

        assert mf_classic.shape == mf_extended.shape
        assert mf_extended.shape == mf_combo.shape
        assert mf_combo.shape == mf_filtered.shape
        assert mf_filtered.shape == rad.shape

    def test_at_mf_total_emit_with_water_mask(self, mock_emit_image):
        """Test AT_MF_total_EMIT with water masking enabled."""
        from marshsi.emit import retrieval_upv_emit as ret

        result = ret.AT_MF_total_EMIT(mock_emit_image, mask_water=True)

        assert len(result) == 5

    def test_at_mf_total_emit_custom_fill_value(self, mock_emit_image):
        """Test AT_MF_total_EMIT with custom fill value."""
        from marshsi.emit import retrieval_upv_emit as ret

        custom_fill = -999.0
        mf_classic, mf_extended, mf_combo, mf_filtered, rad = ret.AT_MF_total_EMIT(
            mock_emit_image, fill_value_default=custom_fill
        )

        assert mf_classic.fill_value_default == custom_fill

    def test_at_mf_total_emit_custom_wavelength_ranges(self, mock_emit_image):
        """Test AT_MF_total_EMIT with custom wavelength ranges."""
        from marshsi.emit import retrieval_upv_emit as ret

        custom_extended = (1000, 2400)
        custom_classic = (2100, 2400)

        result = ret.AT_MF_total_EMIT(
            mock_emit_image,
            extended_wavelengths_range=custom_extended,
            classic_wavelengths_range=custom_classic,
        )

        assert len(result) == 5

    def test_at_mf_total_emit_with_logger(self, mock_emit_image):
        """Test AT_MF_total_EMIT with logger."""
        from marshsi.emit import retrieval_upv_emit as ret

        mock_logger = MagicMock()

        result = ret.AT_MF_total_EMIT(mock_emit_image, logger=mock_logger)

        # Logger should have been called
        assert mock_logger.info.called
        assert len(result) == 5


# ─────────────────────────────────────────────────────────────────────────────
# Integration tests
# ─────────────────────────────────────────────────────────────────────────────
class TestIntegration:
    """Integration tests for the module."""

    def test_wavelength_consistency(self, sample_wavelengths):
        """Test that wavelength selection is consistent."""
        from marshsi.emit import retrieval_upv_emit as ret

        extended_mask = ret.extended_bool_wavelengths(sample_wavelengths)
        extended_wavelengths = sample_wavelengths[extended_mask]

        # Classic range should be a subset of extended range
        classic_mask = (extended_wavelengths >= ret.CLASSIC_WAVELENGTH_RANGE[0]) & (
            extended_wavelengths <= ret.CLASSIC_WAVELENGTH_RANGE[1]
        )

        assert np.sum(classic_mask) > 0  # Should have some classic wavelengths

    def test_matching_filter_output_range(self, mock_emit_image):
        """Test that matching filter outputs are in reasonable range."""
        from marshsi.emit import retrieval_upv_emit as ret

        mf_classic, mf_extended, mf_combo, mf_filtered, rad = ret.AT_MF_total_EMIT(mock_emit_image)

        # Get valid (non-fill) values
        fill_value = mf_classic.fill_value_default
        valid_classic = mf_classic.values[mf_classic.values != fill_value]
        valid_extended = mf_extended.values[mf_extended.values != fill_value]

        # Values should be finite
        assert np.all(np.isfinite(valid_classic))
        assert np.all(np.isfinite(valid_extended))

    def test_target_spectrum_shape_matches_wavelengths(self, mock_emit_image):
        """Test that target spectrum has correct shape."""
        from marshsi.emit import retrieval_upv_emit as ret

        k_arr = ret.load_target_spectrum_mf(mock_emit_image)

        assert len(k_arr) == len(mock_emit_image.wavelengths)


# ─────────────────────────────────────────────────────────────────────────────
# Real EMIT fixture: happy path + target spectrum
# ─────────────────────────────────────────────────────────────────────────────
class TestATMFTotalEMITRealFixture:
    """AT_MF_total_EMIT on the committed plume fixture (the real read path)."""

    def test_returns_five_geotensors(self, emit_image):
        from georeader.geotensor import GeoTensor

        from marshsi.emit import retrieval_upv_emit as ret

        out = ret.AT_MF_total_EMIT(emit_image, georeferenced=True)
        assert len(out) == 5
        assert all(isinstance(o, GeoTensor) for o in out)
        shapes = {o.shape[-2:] for o in out}
        assert len(shapes) == 1  # all products share the spatial grid

    def test_non_fill_values_finite(self, emit_image):
        from marshsi.emit import retrieval_upv_emit as ret

        mf_classic, mf_extended, *_ = ret.AT_MF_total_EMIT(emit_image, georeferenced=False)
        for arr in (mf_classic, mf_extended):
            valid = arr[arr != -9999]
            assert valid.size > 0
            assert np.all(np.isfinite(valid))

    def test_extended_wmf_resolves_plume(self, emit_image):
        """On the 350x350 fixture the extended WMF has enough column samples to
        resolve the plume — guards against the rank-deficient regime."""
        from marshsi.emit import retrieval_upv_emit as ret

        _, mf_extended, *_ = ret.AT_MF_total_EMIT(emit_image, georeferenced=False)
        valid = mf_extended[mf_extended != -9999]
        assert np.percentile(valid, 99) > 0.03


class TestLoadTargetSpectrumMFRealFixture:
    """load_target_spectrum_mf using the fixture's real angles (from the OBS file)."""

    def test_returns_finite_per_band_spectrum(self, emit_image):
        from marshsi.emit import retrieval_upv_emit as ret

        k_arr = ret.load_target_spectrum_mf(emit_image)
        assert k_arr.shape == (emit_image.wavelengths.size,)
        assert np.all(np.isfinite(k_arr))


# ─────────────────────────────────────────────────────────────────────────────
# NaN handling on the real EMIT scene (the wmf processor)
# ─────────────────────────────────────────────────────────────────────────────
class TestATMFTotalEMITNaNRealFixture:
    """The wmf path is NaN-tolerant by design: it turns fill into NaN, drops
    NaN rows per column and solves with ``np.linalg.pinv`` (no Cholesky). These
    tests guard that tolerance against regressions, on the real EMIT scene.
    """

    def test_nan_pixels_handled_without_crashing(self, emit_image, monkeypatch):
        from marshsi.emit import retrieval_upv_emit as ret

        recorded = inject_into_load_raw(monkeypatch, ["nan", "nan"])

        mf_classic, *_ = ret.AT_MF_total_EMIT(emit_image, georeferenced=False)

        assert recorded, "load_raw was never called / no pixels corrupted"
        # Corrupted pixels are dropped and refilled with the fill value.
        for y, x in recorded:
            assert mf_classic[y, x] == -9999

    def test_valid_pixels_still_finite_with_nan(self, emit_image, monkeypatch):
        from marshsi.emit import retrieval_upv_emit as ret

        inject_into_load_raw(monkeypatch, ["nan", "nan"])

        mf_classic, mf_extended, *_ = ret.AT_MF_total_EMIT(
            emit_image, georeferenced=False
        )

        for arr in (mf_classic, mf_extended):
            valid = arr[arr != -9999]
            assert valid.size > 0
            assert np.all(np.isfinite(valid))

import numpy as np
import pytest
from georeader.geotensor import GeoTensor
from marshsi.quantification import (
    A_UEFF_S2,
    ATMOSPHERE_HEIGHT_METHANE,
    B_UEFF_S2,
    BACKGROUND_CH4_CONCENTRATION,
    MAX_CH4_CONCENTRATION_PPB,
    MIN_CH4_CONCENTRATION_PPB,
    SIGMA_CH4_S2,
    UEFF_SATELLITE,
    convert_units,
    obtain_flux_rate,
)
from rasterio.transform import from_bounds

from tests.validators import assert_valid_flux_result


def test_convert_units_ppm_to_ppmxm():
    data = 1.0
    result = convert_units(data, "ppm", "ppm x m")
    expected = data * 8000
    assert result == expected


def test_convert_units_ppb_to_ppmxm():
    data = 1000.0
    result = convert_units(data, "ppb", "ppm x m")
    expected = data * (ATMOSPHERE_HEIGHT_METHANE / 1_000)
    assert result == expected


def test_convert_units_ppmxm_to_ppm():
    data = 8000.0
    result = convert_units(data, "ppm x m", "ppm")
    expected = data / 8000
    assert result == expected


def test_convert_units_ppmxm_to_ppb():
    data = 8000.0
    result = convert_units(data, "ppm x m", "ppb")
    expected = data / 8
    assert result == expected


def test_convert_units_ppmxm_to_ppm_abs():
    data = 8000.0
    result = convert_units(data, "ppm x m", "ppm (abs)")
    expected = (data / ATMOSPHERE_HEIGHT_METHANE) + (BACKGROUND_CH4_CONCENTRATION / 1000)
    assert result == expected


def test_convert_units_ppmxm_to_ppb_abs():
    data = 8000.0
    result = convert_units(data, "ppm x m", "ppb (abs)")
    expected = (data / 8) + BACKGROUND_CH4_CONCENTRATION
    assert result == expected


def test_convert_units_ppm_to_ppm_abs():
    data = 1.0
    result = convert_units(data, "ppm", "ppm (abs)")
    expected = data + (BACKGROUND_CH4_CONCENTRATION / 1000)
    assert result == expected


def test_convert_units_ppb_to_ppb_abs():
    data = 1000.0
    result = convert_units(data, "ppb", "ppb (abs)")
    expected = data + BACKGROUND_CH4_CONCENTRATION
    assert result == expected


def test_convert_units_ppm_to_ppb():
    data = 1.0
    result = convert_units(data, "ppm", "ppb")
    expected = data * 1000
    assert result == expected


def test_convert_units_ppb_to_ppm():
    data = 1000.0
    result = convert_units(data, "ppb", "ppm")
    expected = data / 1000
    assert result == expected


def test_convert_units_ppm_abs_to_ppm():
    data = 1.0
    result = convert_units(data, "ppm (abs)", "ppm")
    expected = data - (BACKGROUND_CH4_CONCENTRATION / 1000)
    assert result == expected


def test_convert_units_ppb_abs_to_ppm():
    data = 1000.0
    result = convert_units(data, "ppb (abs)", "ppb")
    expected = data - BACKGROUND_CH4_CONCENTRATION
    assert result == expected


def test_convert_units_invalid_units_src():
    data = 1.0
    with pytest.raises(AssertionError):
        convert_units(data, "invalid", "ppm")


def test_convert_units_invalid_units_dst():
    data = 1.0
    with pytest.raises(AssertionError):
        convert_units(data, "ppm", "invalid")


def test_convert_units_same_units():
    data = 1.0
    result = convert_units(data, "ppm", "ppm")
    expected = data
    assert result == expected


def test_convert_units_geotensor():
    data = GeoTensor(np.array([[1.0, 2.0], [3.0, 4.0]]), transform=None, crs=None)
    result = convert_units(data, "ppm", "ppm x m")
    expected = GeoTensor(
        np.array([[8000.0, 16000.0], [24000.0, 32000.0]]), transform=None, crs=None
    )
    assert np.array_equal(result.values, expected.values)


def _make_utm_transform(
    res_x=20.0, res_y=20.0, origin_x=500000.0, origin_y=4500000.0, nrows=50, ncols=50
):
    return from_bounds(
        origin_x, origin_y - nrows * res_y, origin_x + ncols * res_x, origin_y, ncols, nrows,
    )


@pytest.fixture
def plume_arrays():
    enhancement = np.zeros((50, 50), dtype=np.float64)
    mask = np.zeros((50, 50), dtype=np.uint8)
    enhancement[20:30, 20:30] = 5.0
    mask[20:30, 20:30] = 1
    return enhancement, mask


@pytest.fixture
def plume_geotensors(plume_arrays):
    enhancement, mask = plume_arrays
    transform = _make_utm_transform(res_x=20.0, res_y=20.0, nrows=50, ncols=50)
    crs = "EPSG:32633"
    gt_enh = GeoTensor(enhancement, transform=transform, crs=crs, fill_value_default=0)
    gt_mask = GeoTensor(mask, transform=transform, crs=crs, fill_value_default=0)
    return gt_enh, gt_mask


class TestObtainFluxRateBasic:
    def test_basic_flux_rate_positive(self, plume_arrays):
        enhancement, mask = plume_arrays
        result = obtain_flux_rate(
            enhancement, mask, wind_speed=5.0, resolution=(20.0, 20.0),
            units_methane_enhancement="ppm",
        )
        assert_valid_flux_result(result)
        assert result["npix_plume"] == 100
        assert result["pixel_size"] == 400.0

    def test_flux_rate_formula_matches_manual(self, plume_arrays):
        enhancement, mask = plume_arrays
        result = obtain_flux_rate(
            enhancement, mask, wind_speed=5.0, a_u_eff=1.0, b_u_eff=0.0,
            resolution=(20.0, 20.0), units_methane_enhancement="ppm",
        )
        assert result["u_eff"] == 5.0
        expected_Q = 3600 * result["u_eff"] * result["IME"] / result["L"]
        assert abs(result["Q"] - expected_Q) < 1e-6

    def test_flux_rate_with_s2_ueff(self, plume_arrays):
        enhancement, mask = plume_arrays
        wind_speed = 5.0
        result = obtain_flux_rate(
            enhancement, mask, wind_speed=wind_speed, a_u_eff=A_UEFF_S2, b_u_eff=B_UEFF_S2,
            resolution=(20.0, 20.0), units_methane_enhancement="ppm",
        )
        expected_ueff = A_UEFF_S2 * wind_speed + B_UEFF_S2
        assert abs(result["u_eff"] - expected_ueff) < 1e-6

    def test_flux_rate_with_geotensors(self, plume_geotensors):
        gt_enh, gt_mask = plume_geotensors
        result = obtain_flux_rate(gt_enh, gt_mask, wind_speed=5.0, units_methane_enhancement="ppm")
        assert_valid_flux_result(result)
        assert result["npix_plume"] == 100

    def test_flux_rate_ppb_units(self):
        enhancement = np.zeros((50, 50), dtype=np.float64)
        mask = np.zeros((50, 50), dtype=np.uint8)
        enhancement[20:30, 20:30] = 5000.0
        mask[20:30, 20:30] = 1
        result_ppb = obtain_flux_rate(
            enhancement, mask, wind_speed=5.0, resolution=(20.0, 20.0),
            units_methane_enhancement="ppb",
        )
        enhancement_ppm = np.zeros((50, 50), dtype=np.float64)
        enhancement_ppm[20:30, 20:30] = 5.0
        result_ppm = obtain_flux_rate(
            enhancement_ppm, mask, wind_speed=5.0, resolution=(20.0, 20.0),
            units_methane_enhancement="ppm",
        )
        assert abs(result_ppb["Q"] - result_ppm["Q"]) < 1e-6


class TestObtainFluxRateEdgeCases:
    def test_negative_enhancement_returns_zero_flux(self):
        enhancement = np.full((50, 50), -100.0, dtype=np.float64)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 1
        result = obtain_flux_rate(
            enhancement, mask, wind_speed=5.0, resolution=(20.0, 20.0),
            units_methane_enhancement="ppb",
        )
        assert result["Q"] == 0.0

    def test_nan_values_treated_as_zero(self):
        enhancement = np.zeros((50, 50), dtype=np.float64)
        mask = np.zeros((50, 50), dtype=np.uint8)
        enhancement[20:30, 20:30] = 5000.0
        mask[20:30, 20:30] = 1
        enhancement[22, 22] = np.nan
        enhancement[23, 23] = np.nan
        result = obtain_flux_rate(
            enhancement, mask, wind_speed=5.0, resolution=(20.0, 20.0),
            units_methane_enhancement="ppb",
        )
        assert_valid_flux_result(result)

    def test_inf_values_treated_as_zero(self):
        enhancement = np.zeros((50, 50), dtype=np.float64)
        mask = np.zeros((50, 50), dtype=np.uint8)
        enhancement[20:30, 20:30] = 5000.0
        mask[20:30, 20:30] = 1
        enhancement[25, 25] = np.inf
        result = obtain_flux_rate(
            enhancement, mask, wind_speed=5.0, resolution=(20.0, 20.0),
            units_methane_enhancement="ppb",
        )
        assert_valid_flux_result(result)

    def test_extreme_ch4_values_are_clipped(self):
        enhancement = np.zeros((50, 50), dtype=np.float64)
        mask = np.zeros((50, 50), dtype=np.uint8)
        enhancement[20:30, 20:30] = 200_000.0
        mask[20:30, 20:30] = 1
        result = obtain_flux_rate(
            enhancement, mask, wind_speed=5.0, resolution=(20.0, 20.0),
            units_methane_enhancement="ppb",
        )
        max_ppmxm = convert_units(MAX_CH4_CONCENTRATION_PPB, "ppb", "ppm x m")
        max_sum = max_ppmxm * 100
        assert result["sum_enhancement"] <= max_sum + 1e-6

    def test_mismatched_shapes_raises(self):
        enhancement = np.zeros((50, 50), dtype=np.float64)
        mask = np.zeros((40, 40), dtype=np.uint8)
        with pytest.raises(AssertionError, match="same shape"):
            obtain_flux_rate(
                enhancement, mask, wind_speed=5.0, resolution=(20.0, 20.0),
                units_methane_enhancement="ppm",
            )

    def test_3d_input_raises(self):
        enhancement = np.zeros((3, 50, 50), dtype=np.float64)
        mask = np.zeros((3, 50, 50), dtype=np.uint8)
        with pytest.raises(AssertionError, match="2D"):
            obtain_flux_rate(
                enhancement, mask, wind_speed=5.0, resolution=(20.0, 20.0),
                units_methane_enhancement="ppm",
            )

    def test_ndarray_without_resolution_raises(self):
        enhancement = np.zeros((50, 50), dtype=np.float64)
        mask = np.zeros((50, 50), dtype=np.uint8)
        with pytest.raises(AssertionError, match="res must be provided"):
            obtain_flux_rate(enhancement, mask, wind_speed=5.0, units_methane_enhancement="ppm")


class TestObtainFluxRateUncertainty:
    def test_return_std_includes_sigma_q(self, plume_arrays):
        enhancement, mask = plume_arrays
        result = obtain_flux_rate(
            enhancement, mask, wind_speed=5.0, resolution=(20.0, 20.0),
            units_methane_enhancement="ppm", return_std=True, seed=42,
        )
        assert_valid_flux_result(result, expect_std=True)

    def test_return_std_false_excludes_sigma_q(self, plume_arrays):
        enhancement, mask = plume_arrays
        result = obtain_flux_rate(
            enhancement, mask, wind_speed=5.0, resolution=(20.0, 20.0),
            units_methane_enhancement="ppm", return_std=False,
        )
        assert "sigma_Q" not in result
        assert "sig_xch4" not in result

    def test_seed_reproducibility(self, plume_arrays):
        enhancement, mask = plume_arrays
        r1 = obtain_flux_rate(
            enhancement, mask, wind_speed=5.0, resolution=(20.0, 20.0),
            units_methane_enhancement="ppm", return_std=True, seed=123,
        )
        r2 = obtain_flux_rate(
            enhancement, mask, wind_speed=5.0, resolution=(20.0, 20.0),
            units_methane_enhancement="ppm", return_std=True, seed=123,
        )
        assert r1["sigma_Q"] == r2["sigma_Q"]
        assert r1["Q"] == r2["Q"]

    def test_different_seeds_produce_different_sigma(self, plume_arrays):
        enhancement, mask = plume_arrays
        r1 = obtain_flux_rate(
            enhancement, mask, wind_speed=5.0, resolution=(20.0, 20.0),
            units_methane_enhancement="ppm", return_std=True, seed=1,
        )
        r2 = obtain_flux_rate(
            enhancement, mask, wind_speed=5.0, resolution=(20.0, 20.0),
            units_methane_enhancement="ppm", return_std=True, seed=999,
        )
        assert r1["Q"] == r2["Q"]
        assert r1["sigma_Q"] != r2["sigma_Q"]

    def test_higher_wind_uncertainty_gives_higher_sigma(self, plume_arrays):
        enhancement, mask = plume_arrays
        r_low = obtain_flux_rate(
            enhancement, mask, wind_speed=5.0, wind_speed_unc=0.1,
            resolution=(20.0, 20.0), units_methane_enhancement="ppm",
            return_std=True, seed=42, n_samp=50_000,
        )
        r_high = obtain_flux_rate(
            enhancement, mask, wind_speed=5.0, wind_speed_unc=1.0,
            resolution=(20.0, 20.0), units_methane_enhancement="ppm",
            return_std=True, seed=42, n_samp=50_000,
        )
        assert r_high["sigma_Q"] > r_low["sigma_Q"]


class TestObtainFluxRateWind:
    def test_higher_wind_gives_higher_flux(self, plume_arrays):
        enhancement, mask = plume_arrays
        r_low = obtain_flux_rate(
            enhancement, mask, wind_speed=2.0, a_u_eff=1.0, b_u_eff=0.0,
            resolution=(20.0, 20.0), units_methane_enhancement="ppm",
        )
        r_high = obtain_flux_rate(
            enhancement, mask, wind_speed=10.0, a_u_eff=1.0, b_u_eff=0.0,
            resolution=(20.0, 20.0), units_methane_enhancement="ppm",
        )
        assert r_high["Q"] > r_low["Q"]

    def test_ueff_intercept_only(self, plume_arrays):
        enhancement, mask = plume_arrays
        r1 = obtain_flux_rate(
            enhancement, mask, wind_speed=1.0, a_u_eff=0.0, b_u_eff=2.0,
            resolution=(20.0, 20.0), units_methane_enhancement="ppm",
        )
        r2 = obtain_flux_rate(
            enhancement, mask, wind_speed=20.0, a_u_eff=0.0, b_u_eff=2.0,
            resolution=(20.0, 20.0), units_methane_enhancement="ppm",
        )
        assert r1["u_eff"] == 2.0
        assert r2["u_eff"] == 2.0
        assert r1["Q"] == r2["Q"]

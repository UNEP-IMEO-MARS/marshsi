"""Tests for mars_mf.lut module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from mars_mf.lut import (
    FILE_LUT_GAS,
    MAX_AMF,
    air_mass_factor,
    load_all_lut,
    read_luts,
)

# --- Helper functions ---


def _create_mock_lut_dataset(
    n_model_wvl: int = 5,
    n_source_wvl: int = 10,
    n_amf: int = 4,
    n_mr: int = 6,
) -> xr.Dataset:
    """Create a mock LUT dataset for testing."""
    wvl_mod = np.linspace(2200, 2400, n_model_wvl)
    wavelength = np.linspace(2100, 2500, n_source_wvl)
    amf_arr = np.linspace(2.0, 4.0, n_amf)
    t_ch4_arr = np.random.uniform(0.8, 1.0, size=(n_model_wvl, n_mr, n_amf))
    mr_ch4_arr = np.random.uniform(1800, 2200, size=(n_mr, n_amf))
    ediff = np.random.uniform(0.1, 0.5, size=n_source_wvl)
    edir = np.random.uniform(0.5, 1.5, size=n_source_wvl)
    trans_tot = np.random.uniform(0.7, 0.95, size=n_source_wvl)

    ds = xr.Dataset(
        {
            "wvl_mod": (["wvl_mod"], wvl_mod),
            "wavelength": (["wavelength"], wavelength),
            "t_ch4_arr": (["wvl_mod", "mr", "amf"], t_ch4_arr),
            "mr_ch4_arr": (["mr", "amf"], mr_ch4_arr),
            "amf_arr": (["amf"], amf_arr),
            "ediff": (["wavelength"], ediff),
            "edir": (["wavelength"], edir),
            "trans_tot": (["wavelength"], trans_tot),
        }
    )
    return ds


class TestAirMassFactor:
    def test_zero_zenith_angles(self):
        amf = air_mass_factor(sza=0.0, vza=0.0)
        assert amf == pytest.approx(2.0)

    def test_symmetric_angles(self):
        amf1 = air_mass_factor(sza=30.0, vza=45.0)
        amf2 = air_mass_factor(sza=45.0, vza=30.0)
        assert amf1 == pytest.approx(amf2)

    def test_45_degree_angles(self):
        amf = air_mass_factor(sza=45.0, vza=45.0)
        expected = 2 / np.cos(np.radians(45.0))
        assert amf == pytest.approx(expected)

    def test_60_degree_sza(self):
        amf = air_mass_factor(sza=60.0, vza=0.0)
        assert amf == pytest.approx(3.0)

    def test_increasing_angles_increase_amf(self):
        amf1 = air_mass_factor(sza=20.0, vza=20.0)
        amf2 = air_mass_factor(sza=40.0, vza=40.0)
        amf3 = air_mass_factor(sza=60.0, vza=60.0)
        assert amf1 < amf2 < amf3

    def test_returns_float(self):
        amf = air_mass_factor(sza=30.0, vza=15.0)
        assert isinstance(amf, float)


class TestLoadAllLut:
    @patch("mars_mf.lut.safe_open_netcdf")
    def test_returns_tuple_of_six_arrays(self, mock_open):
        mock_open.return_value = _create_mock_lut_dataset()
        result = load_all_lut()
        assert isinstance(result, tuple)
        assert len(result) == 6

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_wvl_mod_shape(self, mock_open):
        n_model_wvl = 7
        mock_open.return_value = _create_mock_lut_dataset(n_model_wvl=n_model_wvl)
        wvl_mod, _, _, _, _, _ = load_all_lut()
        assert wvl_mod.shape == (n_model_wvl,)

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_t_ch4_arr_shape_is_transposed(self, mock_open):
        n_model_wvl = 5; n_amf = 4; n_mr = 6
        mock_open.return_value = _create_mock_lut_dataset(n_model_wvl=n_model_wvl, n_amf=n_amf, n_mr=n_mr)
        _, t_ch4_arr, _, _, _, _ = load_all_lut()
        assert t_ch4_arr.shape == (n_amf, n_mr, n_model_wvl)

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_mr_ch4_arr_shape_is_transposed(self, mock_open):
        n_amf = 4; n_mr = 6
        mock_open.return_value = _create_mock_lut_dataset(n_amf=n_amf, n_mr=n_mr)
        _, _, mr_ch4_arr, _, _, _ = load_all_lut()
        assert mr_ch4_arr.shape == (n_amf, n_mr)

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_amf_arr_shape(self, mock_open):
        n_amf = 4
        mock_open.return_value = _create_mock_lut_dataset(n_amf=n_amf)
        _, _, _, amf_arr, _, _ = load_all_lut()
        assert amf_arr.shape == (n_amf,)

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_eg_arr_shape(self, mock_open):
        n_model_wvl = 7
        mock_open.return_value = _create_mock_lut_dataset(n_model_wvl=n_model_wvl)
        _, _, _, _, eg_arr, _ = load_all_lut()
        assert eg_arr.shape == (n_model_wvl,)

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_trans_tot_arr_shape(self, mock_open):
        n_model_wvl = 7
        mock_open.return_value = _create_mock_lut_dataset(n_model_wvl=n_model_wvl)
        _, _, _, _, _, trans_tot_arr = load_all_lut()
        assert trans_tot_arr.shape == (n_model_wvl,)

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_eg_is_sum_of_edir_and_ediff(self, mock_open):
        mock_ds = _create_mock_lut_dataset(n_model_wvl=5, n_source_wvl=100)
        mock_open.return_value = mock_ds
        wvl_mod, _, _, _, eg_arr, _ = load_all_lut()
        assert np.all(eg_arr >= 0)

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_uses_default_file_path(self, mock_open):
        mock_open.return_value = _create_mock_lut_dataset()
        load_all_lut()
        mock_open.assert_called_once_with(FILE_LUT_GAS, cache=False, load=True)

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_uses_custom_file_path(self, mock_open):
        mock_open.return_value = _create_mock_lut_dataset()
        custom_path = "/custom/path/lut.nc"
        load_all_lut(lut_file=custom_path)
        mock_open.assert_called_once_with(custom_path, cache=False, load=True)

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_all_arrays_are_numpy(self, mock_open):
        mock_open.return_value = _create_mock_lut_dataset()
        result = load_all_lut()
        for arr in result:
            assert isinstance(arr, np.ndarray)


class TestReadLuts:
    @patch("mars_mf.lut.safe_open_netcdf")
    def test_returns_tuple_of_three_arrays(self, mock_open):
        mock_open.return_value = _create_mock_lut_dataset()
        result = read_luts(amf=2.5)
        assert isinstance(result, tuple)
        assert len(result) == 3

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_wvl_mod_shape(self, mock_open):
        n_model_wvl = 7
        mock_open.return_value = _create_mock_lut_dataset(n_model_wvl=n_model_wvl)
        wvl_mod, _, _ = read_luts(amf=2.5)
        assert wvl_mod.shape == (n_model_wvl,)

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_t_arr_shape_after_interpolation(self, mock_open):
        n_model_wvl = 5; n_mr = 6
        mock_open.return_value = _create_mock_lut_dataset(n_model_wvl=n_model_wvl, n_mr=n_mr)
        _, t_arr, _ = read_luts(amf=2.5)
        assert t_arr.shape == (n_mr, n_model_wvl)

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_mr_arr_shape_after_interpolation(self, mock_open):
        n_mr = 6
        mock_open.return_value = _create_mock_lut_dataset(n_mr=n_mr)
        _, _, mr_arr = read_luts(amf=2.5)
        assert mr_arr.shape == (n_mr,)

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_uses_default_file_path(self, mock_open):
        mock_open.return_value = _create_mock_lut_dataset()
        read_luts(amf=2.5)
        mock_open.assert_called_once_with(FILE_LUT_GAS, cache=False, load=True)

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_uses_custom_file_path(self, mock_open):
        mock_open.return_value = _create_mock_lut_dataset()
        custom_path = "/custom/path/lut.nc"
        read_luts(amf=2.5, file_lut=custom_path)
        mock_open.assert_called_once_with(custom_path, cache=False, load=True)

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_interpolation_at_exact_amf_value(self, mock_open):
        n_amf = 4
        mock_ds = _create_mock_lut_dataset(n_amf=n_amf)
        mock_open.return_value = mock_ds
        wvl_mod, t_arr, mr_arr = read_luts(amf=2.0)
        assert t_arr is not None
        assert mr_arr is not None

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_interpolation_between_amf_values(self, mock_open):
        mock_ds = _create_mock_lut_dataset(n_amf=4)
        mock_open.return_value = mock_ds
        wvl_mod, t_arr, mr_arr = read_luts(amf=3.0)
        assert t_arr is not None
        assert mr_arr is not None

    @patch("mars_mf.lut.safe_open_netcdf")
    def test_all_arrays_are_numpy(self, mock_open):
        mock_open.return_value = _create_mock_lut_dataset()
        result = read_luts(amf=2.5)
        for arr in result:
            assert isinstance(arr, np.ndarray)


class TestModuleConstants:
    def test_file_lut_gas_is_string(self):
        assert isinstance(FILE_LUT_GAS, str)

    def test_file_lut_gas_ends_with_nc(self):
        assert FILE_LUT_GAS.endswith(".nc")

    def test_max_amf_value(self):
        assert MAX_AMF == 3.92

    def test_max_amf_is_positive(self):
        assert MAX_AMF > 0


class TestIntegration:
    def test_amf_within_typical_range(self):
        amf_min = air_mass_factor(sza=20.0, vza=0.0)
        amf_max = air_mass_factor(sza=60.0, vza=30.0)
        assert 2.0 < amf_min < 2.5
        assert 3.0 < amf_max < 4.0

    def test_amf_symmetry_property(self):
        for sza in [10, 30, 50]:
            for vza in [5, 15, 25]:
                amf1 = air_mass_factor(sza=float(sza), vza=float(vza))
                amf2 = air_mass_factor(sza=float(vza), vza=float(sza))
                assert amf1 == pytest.approx(amf2)

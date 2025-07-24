# This module is a modified version of the Wide Matched Filter (WMF) retrieval method of Roger et al. 2024
# Roger, J., Guanter, L., Gorroño, J., and Irakulis-Loitxate, I.: 
# Exploiting the entire near-infrared spectral range to improve the detection of methane plumes with high-resolution imaging spectrometers, 
# Atmos. Meas. Tech., 17, 1333–1346, https://doi.org/10.5194/amt-17-1333-2024, 2024.

import os

import netCDF4
import numpy as np
from numpy.typing import NDArray
from scipy import interpolate

FILE_LUT_GAS = os.path.join(os.path.dirname(__file__), "output_Tch4_LUT_AMF_VZA_0_v2.nc")

MAX_AMF = 3.92  # Maximum AMF for PRISMA and EnMAP, used to avoid issues with the LUTs

# Adapta el espectro de k a la respuesta espectral de PRISMA. Devuelve espectro de k ajustada a las bandas
def np_where(wvl_M, wvl, wl_resol, bd):
    return np.where(
        np.logical_and(
            wvl_M >= (wvl[bd] - 2.0 * wl_resol[bd]),
            wvl_M <= (wvl[bd] + 2.0 * wl_resol[bd]),
        )
    )


def absolute(wvl, wvl_M, wl_resol, c_arr, bd, li1):
    return np.absolute(wvl[bd] - wvl_M[li1]) / (wl_resol[bd] * c_arr[bd])


def exponential(tmp, exp_arr, bd):
    return np.exp(-(np.power(tmp, exp_arr[bd])))


def norm_s(s):
    return s / np.sum(s)


def calc_jac_rad(mr_gas_arr, n_wvl, t_gas_arr, delta_mr_ref):
    n_pts = len(mr_gas_arr)
    delta_rad = np.zeros((n_pts, n_wvl))
    delta_mr = np.zeros(n_pts)

    for i in range(n_pts):
        delta_rad[i, :] = t_gas_arr[i, :] / t_gas_arr[0, :]
        delta_mr[i] = mr_gas_arr[i] - mr_gas_arr[0]

    n_pol_jac = 2
    jac_gas = np.zeros((n_pol_jac + 1, n_wvl))

    for i in range(n_wvl):
        jac_gas[:, i] = np.polyfit(delta_mr, delta_rad[:, i], n_pol_jac)

    mr_poly = np.array([])
    for i in range(n_pol_jac + 1):
        mr_poly = np.append(mr_poly, [delta_mr**i])

    mr_poly = np.reshape(mr_poly, [n_pol_jac + 1, n_pts])
    mr_poly = np.flip(mr_poly, axis=0)

    jac_spec_rad = np.zeros(n_wvl)
    for i in range(n_wvl):
        jac_spec_rad[i] = 2 * jac_gas[0, i] * delta_mr_ref + jac_gas[1, i]

    return jac_spec_rad


# retrieval_roger_lars
def generate_filter(wvl_M, wvl, wl_resol):
    num_wvl_M = wvl_M.shape[0]
    num_wvl = wvl.shape[0]

    s_norm_M = np.zeros((num_wvl_M, num_wvl))
    exp_max = 2.0
    exp_min = 2.0
    exp_arr = exp_max + (exp_min - exp_max) * np.arange(num_wvl) / (num_wvl - 1)
    c_arr = np.power(1.0 / (np.power(2, exp_arr) * np.log(2.0)), 1.0 / exp_arr)

    for bd in range(num_wvl):
        li1 = np_where(wvl_M, wvl, wl_resol, bd)

        if len(li1[0]) > 0:
            tmp = absolute(wvl, wvl_M, wl_resol, c_arr, bd, li1)
            s = exponential(tmp, exp_arr, bd)
            s_norm_M[li1, bd] = norm_s(s)

    return s_norm_M


def read_luts(amf, file_lut: str = FILE_LUT_GAS):
    nc_lut = netCDF4.Dataset(file_lut, "r", format="NETCDF4")

    wvl_mod = np.array(nc_lut.variables["wvl_mod"])

    tmp = nc_lut.variables["t_ch4_arr"]
    t_arr = np.copy(tmp).T

    tmp = np.array(nc_lut.variables["mr_ch4_arr"])
    mr_arr_all = np.copy(tmp).T

    amf_arr = np.array(nc_lut.variables["amf_arr"])

    f_t = interpolate.interp1d(amf_arr, t_arr, axis=0)
    f_mr = interpolate.interp1d(amf_arr, mr_arr_all, axis=0)

    t_arr = f_t(amf)
    mr_arr = f_mr(amf)

    return wvl_mod, t_arr, mr_arr


def AT_Combo_MF_2(mf_extended, mf_classic):
    std_swir = np.nanstd(mf_extended)
    std_2300 = np.nanstd(mf_classic)

    mf_combo = mf_extended.copy()

    mf_combo[mf_extended > mf_classic] = mf_classic[mf_extended > mf_classic]
    mf_combo[mf_extended <= mf_classic] = mf_combo[mf_extended <= mf_classic] * (
        std_2300 / std_swir
    )

    return mf_combo


# retrieval_roger_lars.py
def AT_MF_select_window_alt(img: NDArray, target_spectrum: NDArray) -> NDArray:
    """
    Vanila matching filter.

    Args:
        img (NDArray): image of shape (H, W, B) where H is the height, W is the width (number of columns) and B is the number of bands.
            Invalid values must be np.nan.
        target_spectrum (NDArray): target spectrum of shape (B,) or (W, B) where B is the number of bands and W is the number of columns.

    Returns:
        NDArray: matching filter of shape (H, W) where H is the height and W is the width (number of columns).
    """

    H, W, B = img.shape

    B_target = target_spectrum.shape[-1]
    assert (
        B == B_target
    ), f"Number of bands in img ({B}) and target_spectrum ({B_target}) must be the same."

    if len(target_spectrum.shape) == 2:
        assert (
            target_spectrum.shape[0] == W
        ), f"Number of columns in img ({W}) and target_spectrum ({target_spectrum.shape[0]}) must be the same."

    MF = np.full((H, W), np.nan)

    for i in range(W):
        a = img[:, i, 0]
        idxs_notnan = np.where(~np.isnan(a))[0]
        size_idxs = len(idxs_notnan)
        ones_array = np.ones((size_idxs, B))

        col_notnan = img[idxs_notnan, i]
        mu = np.nanmean(col_notnan, axis=0)
        mu_array = ones_array * mu

        if len(target_spectrum.shape) == 2:
            t = mu * target_spectrum[i]
        else:
            t = mu * target_spectrum

        try:  # SVD error viene probablemente por una escasa estadistica (muchos NaN). Hacemos la columna = 0.
            cov = np.cov(col_notnan, rowvar=False)
            cov_inv = np.linalg.pinv(cov)

            num = np.dot(np.dot((col_notnan - mu_array), cov_inv), t)
            den = np.dot(np.dot(t, cov_inv), t)

            MF[idxs_notnan, i] = num / den

        except:
            continue

    return MF

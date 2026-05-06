import logging
import os

import numpy as np
from georeader import griddata
from georeader.geotensor import GeoTensor
from georeader.readers import enmap, prisma
from numpy.typing import NDArray
from scipy.signal import medfilt2d
from typing import Optional

from marshsi.matched_filters_upv import (
    AT_Combo_MF_2,
    AT_MF_select_window_alt,
    calc_jac_rad,
    generate_filter,
    read_luts,
    target_spectrum,
    MAX_AMF,
)
from ..lut import air_mass_factor

BAND_NAMES_COMBO = [
    "Rad_out",
    "MF(2100-2400)",
    "WWMF(1000-2400)",
    "MF-Combo",
    "MF(2100-2400)FILTERED",
    "WWMF(1000-2400)FILTERED",
    "MF-Combo-Filtered",
]

# from utils import load_cube, mask_water, save_retrieval
RAD_WAVELENGTH = 2_100


def mask_water(img, img_ref, umbral):
    for i in range(117, 166):
        img[img_ref[:, :, i] < umbral] = np.nan

    return img


def target_spectrum_prisma(pi: prisma.PRISMA, swir_flag: bool) -> NDArray:
    """Compute the methane target signature for a PRISMA scene.

    PRISMA has a SMILE effect: each detector column has a slightly different spectral
    response function (SRF), so the target signature varies across the swath.  This
    function returns a **per-column** array of shape ``(M, B)`` to capture that
    variation, unlike EMIT and EnMAP which return a single 1-D vector.

    Implementation note: the LUT interpolation and Jacobian computation are performed
    *once* for the scene-mean AMF; only ``generate_filter`` is called per column.
    Delegating to :func:`~marshsi.matched_filters_upv.target_spectrum` inside the loop
    would repeat the expensive LUT read *M* times.

    For plume vetting, average the returned array over the column range spanned by the
    plume polygon before passing to ``compute()``.

    Args:
        pi: PRISMA image reader.
        swir_flag: If ``True``, use the SWIR channel (~1000–2440 nm); otherwise VNIR.

    Returns:
        NDArray: Per-column target signatures, shape ``(M, B)``, in ppm⁻1 units
    """
    if swir_flag:
        vza = pi.vza_swir
        sza = pi.sza_swir
        band_array = pi.wavelength_swir
        fwhm_array = pi.fwhm_swir
        N, M, B = pi.ltoa_swir.shape
    else:
        vza = pi.vza_vnir
        sza = pi.sza_vnir
        band_array = pi.wavelength_vnir
        fwhm_array = pi.fwhm_vnir
        N, M, B = pi.ltoa_vnir.shape

    amf = air_mass_factor(sza=sza, vza=vza)
    if amf > MAX_AMF:
        logging.warning(f"AMF exceeds {MAX_AMF}: {amf}, truncated")
        amf = MAX_AMF

    # Compute Jacobian once for the scene-mean AMF; only SRF convolution is per-column.
    wvl_mod, t_gas_arr, mr_gas_arr = read_luts(amf)
    mr_gas_arr = mr_gas_arr / 1000.0
    k_spectre = calc_jac_rad(mr_gas_arr, t_gas_arr, delta_mr_ref=1.0)
    k_array = np.zeros((M, B))
    for i in range(M):
        s = generate_filter(wvl_mod, band_array[i], fwhm_array[i])
        k_array[i] = np.dot(k_spectre, s)

    return k_array


def target_spectrum_enmap(enmapi: enmap.EnMAP) -> NDArray:
    """Compute the methane target signature for an EnMAP scene.

    EnMAP has a uniform SRF across the swath (no significant SMILE effect), so a single
    1-D target spectrum covers all pixels, unlike PRISMA which requires per-column spectra.

    Args:
        enmapi: EnMAP image reader; provides ``sza``, ``vza``, ``wl_center``, and
            ``wl_fwhm`` for the SWIR channel.

    Returns:
        NDArray: Target signature, shape ``(B_swir,)`` where *B_swir* is the number of
            EnMAP SWIR bands, in **ppb-compatible units**.  Multiply by
            ``ATMOSPHERE_HEIGHT_METHANE / 1000`` (~8) to convert to ppm·m.
    """
    amf = air_mass_factor(sza=enmapi.sza, vza=enmapi.vza)
    if amf > MAX_AMF:
        logging.warning(f"AMF exceeds {MAX_AMF}: {amf}, truncated")
        amf = MAX_AMF
    return target_spectrum(amf, enmapi.wl_center["swir"], enmapi.wl_fwhm["swir"])


def MF_sunglint_combo_enmap(enmapi: enmap.EnMAP, logger: Optional[logging.Logger] = None) -> GeoTensor:
    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    img_swir = enmapi.load_product("SPECTRAL_IMAGE_SWIR").values

    # TODO set to NaN invalid values?

    # Switch to channel last
    img_swir = np.transpose(img_swir, (1, 2, 0)).astype(np.float64)

    idx_w_classic = np.arange(90, 127)
    # As Guanter et al. (2021)
    idx_w_swir = np.arange(
        4, 130
    )  # This can be modified to avoid high absorption water vapor bands
    wavelengths_swir = np.array(enmapi.wl_center["swir"])[idx_w_swir]
    wavelengths_classic = np.array(enmapi.wl_center["swir"])[idx_w_classic]

    k_arr_swir = target_spectrum_enmap(enmapi)

    logger.info(
        f"Running WMF in wavelength range: {wavelengths_swir.min()} - {wavelengths_swir.max()}"
    )
    logger.info(
        f"Running MF2100 in wavelength range: {wavelengths_classic.min()} - {wavelengths_classic.max()}"
    )

    mf_swir = AT_MF_select_window_alt(img_swir[..., idx_w_swir], k_arr_swir[idx_w_swir])
    mf_2300 = AT_MF_select_window_alt(img_swir[..., idx_w_classic], k_arr_swir[idx_w_classic])
    mf_combo = AT_Combo_MF_2(mf_extended=mf_swir, mf_classic=mf_2300)

    # replace nan with -1 in all mfs
    mf_swir[np.isnan(mf_swir)] = -1
    mf_2300[np.isnan(mf_2300)] = -1
    mf_combo[np.isnan(mf_combo)] = -1

    cube = np.zeros((7, img_swir.shape[0], img_swir.shape[1]), dtype=np.float32)
    cube[0] = img_swir[..., 91]
    cube[1] = mf_2300
    cube[2] = mf_swir
    cube[3] = mf_combo
    cube[4] = medfilt2d(cube[1], kernel_size=3)
    cube[5] = medfilt2d(cube[2], kernel_size=3)
    cube[6] = medfilt2d(cube[3], kernel_size=3)

    return GeoTensor(cube, crs=enmapi.crs, transform=enmapi.transform, fill_value_default=-1)


def MF_sunglint_combo_prisma(
    pi: prisma.PRISMA,
    logger: Optional[logging.Logger]=None,
    mask_flag: bool = False,
    umbral_mask: float = 0.05,
) -> GeoTensor:
    if logger is None:
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    img_swir = pi.load_raw(swir_flag=True)
    img_vnir = pi.load_raw(swir_flag=False)


    # TODO set invalid values to np.nan??

    if mask_flag:
        img_swir = mask_water(img_swir, img_swir, umbral_mask)
    img_vnir[np.isnan(img_swir[:, :, 117])] = np.nan

    k_arr_vnir = target_spectrum_prisma(pi, swir_flag=False)
    k_arr_swir = target_spectrum_prisma(pi, swir_flag=True)

    img_tot = np.concatenate((img_vnir, img_swir), axis=2).astype(np.float64)
    k_arr_tot = np.concatenate((k_arr_vnir, k_arr_swir), axis=1).astype(np.float64)
    wvl_tot = np.concatenate((pi.wavelength_vnir, pi.wavelength_swir), axis=1)

    wvl_tot_mean = np.mean(wvl_tot, axis=0)

    w_classic = np.arange(117 + 63, 166 + 63)
    w_swir = np.arange(63 + 4, 166 + 63)
    wavelengths_swir = wvl_tot_mean[w_swir]
    wavelengths_classic = wvl_tot_mean[w_classic]
    logger.info(
        f"Running WMF in wavelength range: {wavelengths_swir.min()} - {wavelengths_swir.max()}"
    )
    logger.info(
        f"Running MF2100 in wavelength range: {wavelengths_classic.min()} - {wavelengths_classic.max()}"
    )

    # img = im[:,:,band_idxs_array] #La imagen tiene que estar orientada/ AT==col
    # k_array = k_array[:,band_idxs_array]

    mf_swir = AT_MF_select_window_alt(img_tot[..., w_swir], k_arr_tot[:, w_swir])
    mf_2300 = AT_MF_select_window_alt(img_tot[..., w_classic], k_arr_tot[:, w_classic])
    mf_combo = AT_Combo_MF_2(mf_extended=mf_swir, mf_classic=mf_2300)

    M, N, B = img_tot.shape
    cube = np.zeros((M, N, 7), dtype=np.float32)
    cube[:, :, 0] = img_swir[:, :, 117]
    cube[:, :, 1] = mf_2300
    cube[:, :, 2] = mf_swir
    cube[:, :, 3] = mf_combo
    cube[:, :, 4] = medfilt2d(cube[:, :, 1], kernel_size=3)
    cube[:, :, 5] = medfilt2d(cube[:, :, 2], kernel_size=3)
    cube[:, :, 6] = medfilt2d(cube[:, :, 3], kernel_size=3)

    # transpose to row major
    cube = np.transpose(cube, (1, 0, 2))

    # reproject to UTM 30m resolution
    geotensor = griddata.read_to_crs(
        cube.astype(np.float32), lons=pi.lons, lats=pi.lats, resolution_dst=30
    )
    return geotensor

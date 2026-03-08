import logging
import os
from typing import Optional, Tuple

import numpy as np
from georeader.geotensor import GeoTensor
from georeader.readers import emit
from numpy.typing import NDArray
from scipy.signal import medfilt2d

logger = logging.getLogger(__name__)

from marshsi.matched_filters_upv import (
    AT_Combo_MF_2,
    AT_MF_select_window_alt,
    calc_jac_rad,
    generate_filter,
    read_luts,
    MAX_AMF,
)
from ..lut import air_mass_factor


def load_target_spectrum_mf(emit_image: emit.EMITImage) -> NDArray:
    amf = air_mass_factor(sza=emit_image.mean_sza, vza=emit_image.mean_vza)
    if amf > MAX_AMF:
        logger.warning(f"AMF exceeds {MAX_AMF}: {amf}, truncated")
        amf = MAX_AMF
    wvl_mod, t_gas_arr, mr_gas_arr = read_luts(amf)

    n_wvl = len(wvl_mod)
    mr_gas_arr = mr_gas_arr / 1000.0
    delta_mr_ref = 1.0
    k_spectre = calc_jac_rad(mr_gas_arr, n_wvl, t_gas_arr, delta_mr_ref)
    s = generate_filter(wvl_mod, emit_image.wavelengths, emit_image.fwhm)

    k = np.dot(k_spectre, s)
    return k


EXTENDED_WAVELENGTH_RANGE = (975, 2_445)
CLASSIC_WAVELENGTH_RANGE = (2_100, 2_445)
RAD_WAVELENGTH = 2_100

# PERIODS_EXCLUDE_WATER = [(1350, 1420), (1800, 1945)]
PERIODS_EXCLUDE_WATER = [
    (1260, 1330),
]


def extended_bool_wavelengths(
    wavelengths: NDArray, extended_wavelengths_range=EXTENDED_WAVELENGTH_RANGE
) -> Tuple[np.array, np.array]:
    eb = (wavelengths >= extended_wavelengths_range[0]) & (
        wavelengths <= extended_wavelengths_range[1]
    )

    # remove water absortion bands ranges: (1350-1420) and (1800, 1945)
    for period in PERIODS_EXCLUDE_WATER:
        eb = eb & ~((wavelengths >= period[0]) & (wavelengths <= period[1]))

    return eb


# def AT_MF_total_EMIT(path_img, path_dat_atm, name_img, mask_flag, umbral_mask, save_flag, path_save):

GeoTensorOrNDArray = GeoTensor | NDArray

def AT_MF_total_EMIT(
    emit_image: emit.EMITImage,
    mask_water: bool = False,
    extended_wavelengths_range=EXTENDED_WAVELENGTH_RANGE,
    classic_wavelengths_range=CLASSIC_WAVELENGTH_RANGE,
    water_mask_threshold: float = 0.05,
    fill_value_default: float = -9999,
    georeferenced: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Tuple[GeoTensorOrNDArray, GeoTensorOrNDArray, GeoTensorOrNDArray, GeoTensorOrNDArray, GeoTensorOrNDArray]:
    """
    This function calculates wide matching filter proposed by Roger et al. 2023.
    The wide matching filter

    Args:
        emit (EMITImage): EMITImage object
        mask_water (bool, optional): If True, water pixels are masked. Defaults to False.
        water_mask_threshold (float, optional): Threshold to mask water pixels. Defaults to 0.05.
        fill_value_default (float, optional): Fill value for output products. Defaults to -9999.
        georeferenced (bool, optional): If True, the output products are georreferenced. Defaults to True.
        logger (Optional[logging.Logger], optional): Logger to use. Defaults to None.

    Returns:
        Tuple[GeoTensorOrNDArray, GeoTensorOrNDArray, GeoTensorOrNDArray, GeoTensorOrNDArray, GeoTensorOrNDArray]: Tuple with the following elements:
            - GeoTensorOrNDArray: classic matching filter by column MF(2_150, 2_445)
            - GeoTensorOrNDArray: wide matching filter by column MF(975, 2_445) excluding the water
                    absortion ranges: (1350-1420) and (1800, 1945)
            - GeoTensorOrNDArray: combination of classic and wide matching filters by column.
            - GeoTensorOrNDArray: Filtered wide matching filter by column.
            - GeoTensorOrNDArray: Radiance at 2_150nm in \\muW/(nm cm^2 sr)
    """

    # select wavelengths to use to avoid reading all the wavelengths

    k_arr = load_target_spectrum_mf(emit_image)  # (bands, )

    ex_bool_wv = extended_bool_wavelengths(
        emit_image.wavelengths, extended_wavelengths_range=extended_wavelengths_range
    )

    indexes_extended = np.where(ex_bool_wv)[0]
    k_arr_extended = k_arr[indexes_extended]

    if logger is not None:
        # logger.info(f"Extended wavelengths: {emit_image.wavelengths[indexes_extended]}")
        logger.info(
            f"Extended wavelengths range: [{emit_image.wavelengths[indexes_extended].min()}, {emit_image.wavelengths[indexes_extended].max()}]"
        )

    emit_image_extended_wavelengths = emit_image.read_from_bands(indexes_extended)

    indexes_classic_within_extended = np.where(
        (emit_image_extended_wavelengths.wavelengths >= classic_wavelengths_range[0])
        & (emit_image_extended_wavelengths.wavelengths <= classic_wavelengths_range[1])
    )[0]
    if logger is not None:
        # logger.info(f"Classic wavelengths: {emit_image_extended_wavelengths.wavelengths[indexes_classic_within_extended]}")
        logger.info(
            f"Classic wavelengths range: [{emit_image_extended_wavelengths.wavelengths[indexes_classic_within_extended].min()}, {emit_image_extended_wavelengths.wavelengths[indexes_classic_within_extended].max()}]"
        )

    k_arr_classic = k_arr_extended[indexes_classic_within_extended]

    img = emit_image_extended_wavelengths.load_raw(transpose=False)  # (rows, cols, bands)
    img_masked = img.copy().astype(np.float64)
    img_masked[img_masked == fill_value_default] = np.nan
    if mask_water:
        # 2150-2440nm
        water_mask = np.any(
            img_masked[:, :, indexes_classic_within_extended] < water_mask_threshold,
            axis=-1,
        )  # (rows, cols)
        img_masked[water_mask, :] = np.nan

    # 238 -> 2150nm
    idx_2150_in_indexes_extended = int(
        np.argmin(np.abs(RAD_WAVELENGTH - emit_image_extended_wavelengths.wavelengths))
    )
    rad = img_masked[:, :, idx_2150_in_indexes_extended]

    mf_classic = AT_MF_select_window_alt(
        img_masked[:, :, indexes_classic_within_extended], k_arr_classic
    )
    mf_extended = AT_MF_select_window_alt(img_masked, k_arr_extended)

    mf_combo = AT_Combo_MF_2(mf_extended=mf_extended, mf_classic=mf_classic)

    mf_extended_filtered = medfilt2d(mf_extended, kernel_size=3)

    # fill na values with emit_image.fill_value_default
    mf_extended_filtered = np.nan_to_num(mf_extended_filtered, nan=fill_value_default)
    mf_classic = np.nan_to_num(mf_classic, nan=fill_value_default)
    mf_combo = np.nan_to_num(mf_combo, nan=fill_value_default)
    mf_extended = np.nan_to_num(mf_extended, nan=fill_value_default)

    # georeference
    if not georeferenced:
        return (
            mf_classic,
            mf_extended,
            mf_combo,
            mf_extended_filtered,
            rad,
        )
    
    mf_classic_geotensor = emit_image.georreference(
        mf_classic, fill_value_default=fill_value_default
    ).astype(np.float32)
    mf_extended_geotensor = emit_image.georreference(
        mf_extended, fill_value_default=fill_value_default
    ).astype(np.float32)
    mf_combo_geotensor = emit_image.georreference(
        mf_combo, fill_value_default=fill_value_default
    ).astype(np.float32)
    mf_extended_filtered_geotensor = emit_image.georreference(
        mf_extended_filtered, fill_value_default=fill_value_default
    ).astype(np.float32)
    rad_geotensor = emit_image.georreference(rad, fill_value_default=fill_value_default).astype(
        np.float32
    )

    return (
        mf_classic_geotensor,
        mf_extended_geotensor,
        mf_combo_geotensor,
        mf_extended_filtered_geotensor,
        rad_geotensor,
    )

    # return cube, wvl, k_arr

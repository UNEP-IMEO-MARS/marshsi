"""
References:
 * [Roger 2024] Roger, J., Irakulis-Loitxate, I., Valverde, A., Gorroño, J., Chabrillat, S., Brell, M.,
 & Guanter, L. (2024).
 High-Resolution Methane Mapping With the EnMAP Satellite Imaging Spectroscopy Mission.
 IEEE Transactions on Geoscience and Remote Sensing, 62, 1–12. https://doi.org/10.1109/TGRS.2024.3352403
 * [Thorpe 2023] Thorpe, A. K., Green, R. O., Thompson, D. R., Brodrick, P. G., Chapman, J. W.,
 Elder, C. D., Irakulis-Loitxate, I., Cusworth, D. H., Ayasse, A. K., Duren, R. M.,
 Frankenberg, C., Guanter, L., Worden, J. R., Dennison, P. E., Roberts, D. A.,
 Chadwick, K. D., Eastwood, M. L., Fahlen, J. E., & Miller, C. E. (2023).
 Attribution of individual methane and carbon dioxide emission sources using EMIT observations from space.
 Science Advances, 9(46), eadh2391. https://doi.org/10.1126/sciadv.adh2391
* [Gorroño 2023] Gorroño, J., Varon, D. J., Irakulis-Loitxate, I., & Guanter, L. (2023).
  Understanding the potential of Sentinel-2 for monitoring methane point emissions.
  Atmospheric Measurement Techniques, 16(1), 89–107. https://doi.org/10.5194/amt-16-89-2023
  * [Guanter 2021] Guanter, L., Irakulis-Loitxate, I., Gorroño, J., Sánchez-García, E., Cusworth, D. H.,
 Varon, D. J., Cogliati, S., & Colombo, R. (2021).
 Mapping methane point emissions with the PRISMA spaceborne imaging spectrometer.
 Remote Sensing of Environment, 265, 112671. https://doi.org/10.1016/j.rse.2021.112671
* [Varon 2018] Varon, D. J., Jacob, D. J., McKeever, J., Jervis, D., Durak, B. O. A., Xia, Y.,
  & Huang, Y. (2018).
  Quantifying methane point sources from fine-scale satellite observations of atmospheric methane plumes.
  Atmospheric Measurement Techniques, 11(10), 5673–5686. https://doi.org/10.5194/amt-11-5673-2018
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
from georeader.geotensor import GeoTensor
from numpy.typing import NDArray

# 1800 is the CH4 concentration in ppb of an standard atmosphere
BACKGROUND_CONCENTRATION = BACKGROUND_CH4_CONCENTRATION = 1_800  # ppb
MAX_CH4_CONCENTRATION_LUT = 16_000  # ppb

ATMOSPHERE_HEIGHT_METHANE = 8_000  # m
SIGMA_CH4_PRISMA_DEFAULT_PPB = 140.0  # ppb
SIGMA_CH4_ENMAP_DEFAULT_PPB = 140.0  # ppb

SIGMA_CH4_S2_PPB = 205.64270005117675
SIGMA_CH4_S2 = {
    "ppb": SIGMA_CH4_S2_PPB,
    "ppm": SIGMA_CH4_S2_PPB / 1_000,
    "ppm x m": SIGMA_CH4_S2_PPB / 1_000 * ATMOSPHERE_HEIGHT_METHANE,
}

SIGMA_CH4_MSAT_PPB = 100.0
SIGMA_CH4_MSAT = {
    "ppb": SIGMA_CH4_MSAT_PPB,
    "ppm": SIGMA_CH4_MSAT_PPB / 1_000,
    "ppm x m": SIGMA_CH4_MSAT_PPB / 1_000 * ATMOSPHERE_HEIGHT_METHANE,
}

A_UEFF_S2 = 0.33  # [Varon 2018]
B_UEFF_S2 = 0.45  # [Varon 2018]
MAX_CH4_CONCENTRATION_PPB = 100_000  # ppb
MIN_CH4_CONCENTRATION_PPB = -600  # ppb

EMIT_UNCERTAINTY_MAG1C_DEFAULT = 400.0  # ppm x m
EMIT_UNCERTAINTY_MAG1C_DEFAULT_PPB = (
    EMIT_UNCERTAINTY_MAG1C_DEFAULT / ATMOSPHERE_HEIGHT_METHANE * 1000
)

# Constant dictionary with the standard deviation of the methane retrieval for different satellites in ppb
# We use a constant value; nevertheless the better approach would be to estimate the noise on each retrieval.
SIGMA_CH4_SATELLITE_PPB = {
    "S2A": SIGMA_CH4_S2_PPB,
    "S2B": SIGMA_CH4_S2_PPB,
    "S2C": SIGMA_CH4_S2_PPB,
    "LC08": SIGMA_CH4_S2_PPB,
    "LC09": SIGMA_CH4_S2_PPB,
    "LE07": SIGMA_CH4_S2_PPB,
    "LT05": SIGMA_CH4_S2_PPB,
    "LT04": SIGMA_CH4_S2_PPB,
    "EMIT": EMIT_UNCERTAINTY_MAG1C_DEFAULT_PPB,
    "PRISMA": SIGMA_CH4_PRISMA_DEFAULT_PPB,
    "EnMAP": SIGMA_CH4_ENMAP_DEFAULT_PPB,
    "MSAT": SIGMA_CH4_MSAT_PPB,
    "NOAA20": SIGMA_CH4_MSAT_PPB,
    "NOAA21": SIGMA_CH4_MSAT_PPB,
    "SNPP": SIGMA_CH4_MSAT_PPB,
    "S3B": SIGMA_CH4_MSAT_PPB,
    "S3A": SIGMA_CH4_MSAT_PPB,
    "S5P": SIGMA_CH4_MSAT_PPB,
    "MethaneAIR": SIGMA_CH4_MSAT_PPB,
}

UEFF_SATELLITE = {
    "S2A": (A_UEFF_S2, B_UEFF_S2),  # [Varon 2018]
    "S2B": (A_UEFF_S2, B_UEFF_S2),  # [Varon 2018]
    "S2C": (A_UEFF_S2, B_UEFF_S2),  # [Varon 2018]
    "LC08": (A_UEFF_S2, B_UEFF_S2),  # [Varon 2018]
    "LC09": (A_UEFF_S2, B_UEFF_S2),  # [Varon 2018]
    "LE07": (A_UEFF_S2, B_UEFF_S2),  # [Varon 2018]
    "LT05": (A_UEFF_S2, B_UEFF_S2),  # [Varon 2018]
    "LT04": (A_UEFF_S2, B_UEFF_S2),  # [Varon 2018]
    "EMIT": (0.36, 0.66),  # Calibration provided by Michael Steiner (IMEO-619)
    # [Thorpe 2023] -> (1, 0)
    "PRISMA": (0.34, 0.44),  # [Guanter 2021]
    "EnMAP": (0.34, 0.44),  # [Roger et al 2024]
    "MethaneAIR": (0.25, 0.81),  # Calibration provided by Michael Steiner (IMEO-533)
    "MSAT": (1, 0),
    "NOAA20": (1, 0),
    "NOAA21": (1, 0),
    "SNPP": (1, 0),
    "S3B": (1, 0),
    "S3A": (1, 0),
    "S5P": (1, 0),
}

VALID_UNITS = ["ppm", "ppb", "ppm x m", "ppm (abs)", "ppb (abs)", "ppm x m (abs)"]


def convert_units(
    data: Union[NDArray, GeoTensor, float], units_src: str, units_dst: str
) -> Union[NDArray, GeoTensor, float, int]:
    """
    Converts the units of the data from `units_src` to `units_dst`.

    Args:
        data (Union[NDArray, GeoTensor, float, int]): data to convert
        units_src (str): one of "ppm", "ppb", "ppm x m", "ppm (abs)", "ppb (abs)", or "ppm x m (abs)"
        units_dst (str): one of "ppm", "ppb", "ppm x m", "ppm (abs)", "ppb (abs)", or "ppm x m (abs)"

    Raises:
        ValueError: if `units_src` or `units_dst` are not one of "ppm", "ppb", "ppm x m", "ppm (abs)", "ppb (abs)", or "ppm x m (abs)"

    Returns:
        Union[NDArray, GeoTensor, float]: converted data
    """
    msg = "Incorrect Units."
    msg += f"\nMust Be one of: {VALID_UNITS}"
    msg += f"\nGiven Units: {units_src}"
    assert units_src in VALID_UNITS, msg
    msg = "Incorrect Units."
    msg += f"\nMust Be one of: {VALID_UNITS}"
    msg += f"\nGiven Units: {units_dst}"
    assert units_dst in VALID_UNITS, msg

    if units_src == units_dst:
        return data

    # Handle absolute units
    if "(abs)" in units_src:
        units_src = units_src.replace(" (abs)", "")
        data = data - convert_units(BACKGROUND_CH4_CONCENTRATION, "ppb", units_src)

    factor_to_ppmxm = 1
    if units_src == "ppm x m":
        pass
    elif units_src == "ppm":
        factor_to_ppmxm = ATMOSPHERE_HEIGHT_METHANE
    elif units_src == "ppb":
        factor_to_ppmxm = ATMOSPHERE_HEIGHT_METHANE / 1000
    else:
        raise ValueError(f"units_src must be one of 'ppm', 'ppb', or 'ppm x m' found {units_src}")

    # Strip (abs) suffix for conversion checks, but remember if we need to add background
    add_background = "(abs)" in units_dst
    units_dst_base = units_dst.replace(" (abs)", "") if add_background else units_dst

    if "ppm x m" == units_dst_base:
        data = data * factor_to_ppmxm
    if "ppm" == units_dst_base:
        data = data * (factor_to_ppmxm / ATMOSPHERE_HEIGHT_METHANE)
    elif "ppb" == units_dst_base:
        data = data * (factor_to_ppmxm * 1000 / ATMOSPHERE_HEIGHT_METHANE)

    if add_background:
        data = data + convert_units(BACKGROUND_CH4_CONCENTRATION, "ppb", units_dst_base)

    return data


def obtain_flux_rate(
    methane_enhancement_image: Union[NDArray, GeoTensor],
    plume_mask_binary: Union[NDArray, GeoTensor],
    wind_speed: float,
    a_u_eff: float = 1,
    b_u_eff: float = 0,
    a_std: float = 0.01,
    b_std: float = 0.01,
    wind_speed_unc: float = 0.5,
    n_samp: int = 100_000,
    sig_xch4: Optional[float] = None,
    return_std: bool = False,
    seed: Optional[int] = None,
    resolution: Optional[Tuple[float, float]] = None,
    units_methane_enhancement: str = "ppm",
) -> Dict[str, Union[float, int]]:
    """
    Calculates the flux rate of a methane plume given the methane enhancement image, the binary plume mask, and the wind speed.

    This function assumes that methane_enhancement_image is is UTM coordinates (i.e. in meters).

    Effective wind speed for Sentinel-2:
        0.33 * wind_speed + 0.45 Based on 3.5m/s (Cusworth 2019) used for simulations and ueff model in Varon 2020.

    Adapted from uncertainty propagation code of Javier Gorrono 30th June 2022 (adapted from original Luis Guanter)

    Args:
        methane_enhancement_image (Union[NDArray, GeoTensor]): in `units_methane_enhancement` that could be ppm, ppb, or ppm x m
        plume_mask_binary (Union[NDArray, GeoTensor]): binary mask of the plume
        wind_speed (float): wind speed in m/s.
        a_u_eff (float, optional): Slope of the linear relation between wind speed and effective wind speed. Defaults to 1.
        b_u_eff (float, optional): Intercept of the linear relation between wind speed and effective wind speed. Defaults to 0.
        a_std (float, optional): Standard deviation of the slope of the linear relation between wind speed and effective wind speed. Defaults to 0.01.
        b_std (float, optional): Standard deviation of the intercept of the linear relation between wind speed and effective wind speed. Defaults to 0.01.
        wind_speed_unc (float, optional): Relative error of the wind speed in m/s.
            Defaults to 0.5.
        n_samp (int, optional): Number of samples for MonteCarlo propagation. Defaults to 100000.
        sig_xch4 (float, optional): Standard deviation of the methane enhancement retrieval. In `units_methane_enhancement`.
            This value is satellite specific.
        return_std (bool, optional): If True, returns the standard deviation of the flux rate. Defaults to False.
        seed (Optional[int], optional): Seed for the random number generator. Defaults to None.
        resolution (Optional[Tuple[float,float]], optional): Resolution of the methane enhancement image in meters.
            This is used only if the image is not a GeoTensor. Defaults to None.
        units_methane_enhancement (str): One of "ppm", "ppb", or "ppm x m"

    Returns:
        Dict[str, Union[float,int]]: dictionary with the following fields:
            - Q: flux rate in kg/h
            - L: length of the plume estimated as sqrt(npix_plume * np.prod(plume_mask_binary.res)) in m
            - npix_plume: number of pixels in the plume
            - IME: ime in kg
            - sum_enhancement: sum of the methane enhancement in the plume in ppm x m
            - u_eff: effective wind speed in m/s
            - pixel_size: pixel size in m²
            - sigma_q: standard deviation of the flux rate in kg/h if `return_std` is True
            - sig_xch4: standard deviation of the methane enhancement retrieval in ppm x m if `return_std` is True
    """

    assert methane_enhancement_image.shape == plume_mask_binary.shape, (
        f"methane_enhancement_image and plume_mask_binary must have the same shape {methane_enhancement_image.shape} != {plume_mask_binary.shape}"
    )
    assert len(methane_enhancement_image.shape) == 2, (
        f"methane_enhancement_image and plume_mask_binary must be 2D images {methane_enhancement_image.shape}"
    )

    if isinstance(methane_enhancement_image, GeoTensor):
        assert isinstance(plume_mask_binary, GeoTensor), (
            "methane_enhancement_image and plume_mask_binary must be both GeoTensor or both np.ndarray"
        )
        assert methane_enhancement_image.same_extent(plume_mask_binary), (
            f"methane_enhancement_image and plume_mask_binary must have the same extent {methane_enhancement_image.transform} != {plume_mask_binary.transform}"
        )
        resolution = methane_enhancement_image.res
        methane_enhancement_image = methane_enhancement_image.copy()
        methane_enhancement_image_values = methane_enhancement_image.values
        methane_enhancement_image.values[
            methane_enhancement_image.values == methane_enhancement_image.fill_value_default
        ] = 0
        plume_mask_binary_values = plume_mask_binary.values
    else:
        assert resolution is not None, (
            "res must be provided if methane_enhancement_image is a np.ndarray"
        )
        assert len(resolution) == 2, "res must be a tuple with two elements"
        methane_enhancement_image_values = methane_enhancement_image
        plume_mask_binary_values = plume_mask_binary

    if sig_xch4 is None:
        sig_xch4 = SIGMA_CH4_S2[units_methane_enhancement]

    max_ch4_concentration = convert_units(
        MAX_CH4_CONCENTRATION_PPB, "ppb", units_methane_enhancement
    )

    # Allow negative values in methane enhancement up to -600 ppb (IMEO-621)
    min_ch4_concentration = convert_units(
        MIN_CH4_CONCENTRATION_PPB, "ppb", units_methane_enhancement
    )

    # Replace NaN and inf values by 0 (MAIR and MSAT images have many NaN values)
    methane_enhancement_image_values = np.nan_to_num(
        methane_enhancement_image_values, nan=0.0, posinf=0.0, neginf=0.0
    )

    methane_enhancement_image_values = np.clip(
        methane_enhancement_image_values, min_ch4_concentration, max_ch4_concentration
    )

    # convert quantities to ppm x m
    methane_enhancement_image_values = convert_units(
        methane_enhancement_image_values, units_methane_enhancement, "ppm x m"
    )
    sig_xch4 = convert_units(sig_xch4, units_methane_enhancement, "ppm x m")

    # standard deviation should always be positive
    sig_xch4 = np.clip(sig_xch4, a_min=0, a_max=None)

    # Code of Itziar (adapted from Javier Gorrono and Luis Guanter)
    # self.ime = 8000 * (np.sum(self.delta_mrch4[plume_ind] / 1000) * pix_res * pix_res * 1000 * 0.01604) / (
    #             1000000 * 22.4)
    # ueff = 0.33 * self.u10 + 0.45  # Based on 3.5m/s (Cusworth 2019) used for simulations and ueff model in Varon 2020.
    # ueff = 0.23 * self.u10 + 0.73  # fitting at 10t/hr
    # self.q = 3600 * ueff * self.ime / np.sqrt(np.sum(plume_ind) * pix_res * pix_res)

    # Conversion using Thorpe formula (see slide 9 https://unitednations.sharepoint.com/:p:/r/sites/IMEOAnalytics/Shared%20Documents/General/plume_entity_and_quantification.pptx?d=w6896cc5d6a394b19b728f07db99ca3b5&csf=1&web=1&e=bqeR5c)
    if not plume_mask_binary_values.dtype == bool:
        binary_mask = plume_mask_binary_values != 0
    else:
        binary_mask = plume_mask_binary_values

    sum_enhancement = np.sum(methane_enhancement_image_values[binary_mask])
    ime = (sum_enhancement * np.prod(resolution) * 1_000 * 0.01604) / (1e6 * 22.4)

    npix_plume = np.sum(binary_mask)

    L = np.sqrt(npix_plume * np.prod(resolution))

    effective_wind_speed = a_u_eff * wind_speed + b_u_eff

    sum_enhancement = float(sum_enhancement)
    # Avoid negative fluxes, the enhancement can be negative due to noise
    if sum_enhancement > 0:
        Q = 3600 * effective_wind_speed * ime / L
        Q = float(Q)
    else:
        Q = 0.0

    out_dict = {
        "Q": Q,
        "L": float(L),
        "npix_plume": int(npix_plume),
        "IME": float(ime),
        "sum_enhancement": sum_enhancement,
        "pixel_size": float(np.prod(resolution)),
        "u_eff": float(effective_wind_speed),
    }
    if not return_std:
        return out_dict

    # Uncertainty propagation
    # sig_xch4_ppmm_arr = sig_xch4 * 10000. / 1250  # converts from ppb to ppm x m

    # Use a seeded random number generator to obtain reproducible flux rate uncertainties
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # https://stats.stackexchange.com/questions/164505/standard-error-for-sum
    sig_IME = (
        float(sig_xch4) * float(np.sqrt(npix_plume)) * float(np.prod(resolution)) * 1_000 * 0.01604
    ) / (1e6 * 22.4)

    # wind_speed_n = rng.normal(1, wind_speed_unc, n_samp)

    noise_arr = rng.normal(0, 1, n_samp)

    # MonteCarlo propagation of the U10 to Ueff
    wind_speed_n = wind_speed * (
        1 + noise_arr * wind_speed_unc
    )  # uncertainty proportional to the wind speed because it is relative error
    # wind_speed_n = rng.normal(wind_speed, wind_speed_unc*wind_speed, n_samp) # equivalent to the previous line

    ueff_n = wind_speed_n * rng.normal(a_u_eff, a_std, n_samp) + rng.normal(b_u_eff, b_std, n_samp)

    # MonteCarlo propagation of the Ueff and IME to Q
    sigma_q = np.std(3600.0 * ueff_n * rng.normal(ime, sig_IME, n_samp) / L)
    out_dict["sigma_Q"] = sigma_q
    out_dict["sig_xch4"] = sig_xch4

    return out_dict


"""
#!/usr/bin/env python
# coding: utf-8
# Javier Gorrono 30th June 2022 (adapted from original Luis Guanter)

import numpy as np

# DEFINITION  OF PARAMETERS  -------------------------------------------------------------------------------------------
a = 0.33  # Ueff = a*u10 + b
b = 0.45
# Here a and b will be assumed uncorrelated although they have a small correlation of 0.28
a_std = 0.01  # uncertainty of slope based on simulation sensitivity
b_std = 0.01  # uncertainty of intercept based on simulation sensitivity

pix_size = 20  # S2 pixel size in meters
u10_unc = 0.5  # assumes a 50% uncertainty on the wind speed
n_samp = 100000  # number of samples for MonteCarlo propagation

#  INPUT VALUES THAT YOU NEED TO SET  ----------------------------------------------------------------------------------
sig_xch4 = 205.64270005117675  # ppb, from simulations
ime = 766.286645300642
L = 515.363949069005
u10 = 1.18491324
# ----------------------------------------------------------------------------------------------------------------------

npix_arr = L ** 2 / (pix_size * pix_size)
sig_xch4_ppmm_arr = sig_xch4 * 10000. / 1250  # converts from ppb to ppm*m

# 10000. / 1250 = 8 => equivalent to multiplying by 8_000 and divide by 1_000

# https://stats.stackexchange.com/questions/164505/standard-error-for-sum
sig_IME = np.sqrt(npix_arr) * sig_xch4_ppmm_arr * 1.e-6 * pix_size * pix_size * 1000 * (1 / 22.4) * 0.01604

np.random.seed(1)  # para resultados repetibles
noise_arr = np.random.normal(0, 1, n_samp)
# MonteCarlo propagation of the U10 to Ueff
u10_n = u10 * (1 + noise_arr[:] * u10_unc)
ueff_n = u10_n * np.random.normal(a, a_std, len(u10_n)) + np.random.normal(b, b_std,len(u10_n))
# MonteCarlo propagation of the Ueff and IME to Q
sigma_q = np.std(3600. * np.random.normal(ime, sig_IME, n_samp) * ueff_n / L)
q_arr = 3600. * ime * (u10 * a + b) / L

print(np.round(np.array(q_arr), 2))
print(np.round(np.array(sigma_q), 2))

    """

"""
Unified module for reading LUT (Look-Up Table) files for methane retrieval.

This module centralizes all LUT file reading operations. It handles the loading
of pre-calculated Radiative Transfer Model (RTM) outputs, specifically:
1.  **Methane Transmittance (T_CH4)**: The atmospheric transmission due to methane absorption.
2.  **Solar Irradiance (Eg)**: The total downwelling solar flux (Direct + Diffuse).
3.  **Mixing Ratios**: The concentration of Methane in the atmospheric column.

Units assumed throughout:
- Wavelengths: Nanometers [nm]
- Irradiance: Watts per square meter per nanometer [mW·m⁻²·nm⁻¹]
- Mixing Ratios: Parts per million [ppm]
- Transmittance: Unitless [0-1]
- Air Mass Factor (AMF): Unitless geometric factor

Sentinel-2 and Landsat-8/9 notebook:
https://wedevelop.unep.org/projects/OGMP/repos/marsml-notebooks/browse/notebooks/mlretrieval/luts_plot.ipynb
https://wedevelop.unep.org/projects/OGMP/repos/marsml-notebooks/browse/notebooks/paper/explorelut.ipynb

"""

import os
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate

from georeader.io import safe_open_netcdf

# Default LUT file path
FILE_LUT_GAS = os.path.join(os.path.dirname(__file__), "output_Tch4_LUT_AMF_VZA_0_v2.nc")

# Maximum AMF for PRISMA and EnMAP, used to avoid issues with the LUTs
MAX_AMF = 3.92


def load_all_lut(
    lut_file: str = FILE_LUT_GAS,
) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Loads all variables from the LUT file and computes total global irradiance.

    This function reads raw Radiative Transfer Model outputs. It computes the
    Global Irradiance (Eg) by summing the direct (Edir) and diffuse
    (Ediff) components:

    Eg(λ) = Edir(λ) + Ediff(λ)

    It then spectrally resamples (interpolates) these high-resolution irradiance
    grids onto the coarser model wavelength grid used for retrieval.

    Args:
        lut_file (str): LUT file path. Defaults to FILE_LUT_GAS.

    Returns:
        Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray]:

            1. **wvl_mod**: Model wavelength grid center definitions.
               Shape: (n_model_wavelengths,) | Unit: [nm]

            2. **t_ch4_arr**: Methane transmittance LUT.
               Shape: (n_air_mass_factors, n_methane_ratios, n_model_wavelengths) | Unit: [unitless]

            3. **mr_ch4_arr**: Methane mixing ratios LUT.
               Shape: (n_air_mass_factors, n_methane_ratios) | Unit: [ppm]

            4. **amf_arr**: Air Mass Factor coordinate grid (geometric path length enhancement).
               Shape: (n_air_mass_factors,) | Unit: [unitless]

            5. **eg_arr**: Total global irradiance (Eg) interpolated to model grid.
               Shape: (n_model_wavelengths,) | Unit: [W·m⁻²·nm⁻¹]

            6. **trans_tot_arr**: Total atmospheric transmission interpolated to model grid.
               Shape: (n_model_wavelengths,) | Unit: [unitless]
    """
    ds = safe_open_netcdf(lut_file, cache=False, load=True)

    # --- Model Grid Definition ---

    # Shape: (n_model_wavelengths,)
    # Physical: The wavelength centers for the retrieval forward model.
    # Unit: [nm]
    wvl_mod = np.array(ds["wvl_mod"].values)

    # --- Methane Radiative Transfer LUTs (Transposed) ---

    # Raw Shape: (n_model_wavelengths, n_methane_ratios, n_air_mass_factors)
    # Transformation: Transposed (.T) to align AMF as the primary interpolation axis.
    # Final Shape: (n_air_mass_factors, n_methane_ratios, n_model_wavelengths)
    # Physical: Pre-computed transmittance spectra. Governed by Beer-Lambert Law:
    # Equation: T(λ) = exp(-τ_ch4(λ)) = exp(-σ(λ) · N_col · AMF)
    # Unit: [unitless] (0.0 to 1.0)
    t_ch4_arr = np.array(ds["t_ch4_arr"].values).T

    # Raw Shape: (n_methane_ratios, n_air_mass_factors)
    # Transformation: Transposed (.T)
    # Final Shape: (n_air_mass_factors, n_methane_ratios)
    # Physical: The effective mixing ratios corresponding to the transmittance table.
    # Equation: X_ch4 = n_ch4 / n_air_total
    # Unit: [ppm]
    mr_ch4_arr = np.array(ds["mr_ch4_arr"].values).T

    # Shape: (n_air_mass_factors,)
    # Physical: The geometric Air Mass Factors: 1/cos(θ_sun) + 1/cos(θ_view)
    # Unit: [unitless]
    amf_arr = np.array(ds["amf_arr"].values)

    # --- Solar Irradiance and Transmission (High Res Source) ---

    # Shape: (n_source_wavelengths,)
    # Physical: High-spectral-resolution grid from the solar reference spectrum.
    # Unit: [nm]
    wavelength = np.array(ds["wavelength"].values)

    # Shape: (n_source_wavelengths,)
    # Physical: Diffuse component of downwelling irradiance (Ediff).
    # Unit: [W·m⁻²·nm⁻¹]
    ediff = np.array(ds["ediff"].values)

    # Shape: (n_source_wavelengths,)
    # Physical: Direct component of downwelling irradiance (Edir).
    # Unit: [W·m⁻²·nm⁻¹]
    edir = np.array(ds["edir"].values)

    # Shape: (n_source_wavelengths,)
    # Physical: Total atmospheric transmission on source grid (Rayleigh + Aerosol + Gas).
    # Equation: T_tot(λ) = T_rayleigh(λ) × T_aerosol(λ) × T_gas(λ)
    # Unit: [unitless]
    trans_tot = np.array(ds["trans_tot"].values)

    ds.close()

    # --- Radiative Calculation & Spectral Resampling ---

    # Equation: Eg = Ediff + Edir
    # Operation: Element-wise addition summing diffuse and direct flux.
    # Shape: (n_source_wavelengths,)
    # Unit: [W·m⁻²·nm⁻¹]
    eg = ediff + edir

    # Interpolators: Create mapping functions f(λ) -> Value
    # Domain: High-resolution source wavelengths (n_source_wavelengths)
    eg_interp = interpolate.interp1d(wavelength, eg, kind="cubic", bounds_error=False, fill_value=0)
    trans_tot_interp = interpolate.interp1d(
        wavelength, trans_tot, kind="cubic", bounds_error=False, fill_value=0
    )

    # Apply Interpolation: Resample physics from Source Grid -> Model Grid

    # Final Shape: (n_model_wavelengths,)
    # Physical: Global Irradiance resampled to retrieval wavelengths.
    eg_arr = eg_interp(wvl_mod)

    # Final Shape: (n_model_wavelengths,)
    # Physical: Total Transmission resampled to retrieval wavelengths.
    trans_tot_arr = trans_tot_interp(wvl_mod)

    return wvl_mod, t_ch4_arr, mr_ch4_arr, amf_arr, eg_arr, trans_tot_arr


def read_luts(amf: float, file_lut: str = FILE_LUT_GAS) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Interpolates Methane LUTs to a specific observation geometry (AMF).

    This function adjusts the generic LUTs to the specific geometric path length
    of the current observation. It performs linear interpolation along the AMF axis.

    The effective optical path is defined by the Air Mass Factor (AMF):
    AMF = 1/cos(θ_SZA) + 1/cos(θ_VZA)

    Args:
        amf (float): The specific Air Mass Factor for the observation.
        file_lut (str): Path to LUT file. Defaults to FILE_LUT_GAS.

    Returns:
        Tuple[NDArray, NDArray, NDArray]:

            1. **wvl_mod**: Model wavelength array.
               Shape: (n_model_wavelengths,) | Unit: [nm]

            2. **t_arr**: Transmittance spectrum interpolated to the specific AMF.
               Shape: (n_methane_ratios, n_model_wavelengths) | Unit: [unitless]

            3. **mr_arr**: Mixing ratios corresponding to the transmittance curves, interpolated to AMF.
               Shape: (n_methane_ratios,) | Unit: [ppm]
    """
    ds = safe_open_netcdf(file_lut, cache=False, load=True)

    # Shape: (n_model_wavelengths,)
    wvl_mod = np.array(ds["wvl_mod"].values)

    # Shape: (n_air_mass_factors, n_methane_ratios, n_model_wavelengths)
    # Note: Transposed. Axis 0 represents the AMF dimension we will collapse via interpolation.
    t_arr_full = np.array(ds["t_ch4_arr"].values).T

    # Shape: (n_air_mass_factors, n_methane_ratios)
    # Note: Transposed. Axis 0 represents the AMF dimension.
    mr_arr_all = np.array(ds["mr_ch4_arr"].values).T

    # Shape: (n_air_mass_factors,)
    amf_arr = np.array(ds["amf_arr"].values)

    ds.close()

    # --- Interpolation Setup ---

    # We create interpolators along Axis 0 (AMF) to map generic geometry -> specific observation geometry.
    # Method: Linear Interpolation
    # Equation: y = y₀ + (x - x₀) · (y₁ - y₀) / (x₁ - x₀)

    # Input X: amf_arr (n_air_mass_factors,)
    # Input Y: t_arr_full (n_air_mass_factors, n_methane_ratios, n_model_wavelengths)
    f_t = interpolate.interp1d(amf_arr, t_arr_full, axis=0)

    # Input X: amf_arr (n_air_mass_factors,)
    # Input Y: mr_arr_all (n_air_mass_factors, n_methane_ratios)
    f_mr = interpolate.interp1d(amf_arr, mr_arr_all, axis=0)

    # --- Interpolation Execution ---

    # Operation: Evaluate interpolator at scalar `amf`.
    # Dimensionality Change: (n_air_mass_factors, ...) -> (...)

    # Final Shape: (n_methane_ratios, n_model_wavelengths)
    # Physical: The family of transmittance curves valid for this specific geometric path.
    t_arr = f_t(amf)

    # Final Shape: (n_methane_ratios,)
    # Physical: The mixing ratios associated with the rows of t_arr for this geometry.
    mr_arr = f_mr(amf)

    return wvl_mod, t_arr, mr_arr


def air_mass_factor(sza: float, vza: float) -> float:
    """
    Air Mass Factor (AMF) for a given Solar Zenith Angle (SZA) and Viewing Zenith Angle (VZA).
    The AMF is given by the formula:
    AMF = 1 / cos(VZA) + 1 / cos(SZA)

    Args:
        sza (float): Solar Zenith Angle in degrees
        vza (float): Viewing Zenith Angle in degrees

    Returns:
        float: Air Mass Factor
    """
    return 1.0 / np.cos(np.radians(vza)) + 1.0 / np.cos(np.radians(sza))

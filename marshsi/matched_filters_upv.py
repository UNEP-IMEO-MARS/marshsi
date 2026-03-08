# This module is a modified version of the Wide Matched Filter (WMF) retrieval method of Roger et al. 2024
# Roger, J., Guanter, L., Gorroño, J., and Irakulis-Loitxate, I.: 
# Exploiting the entire near-infrared spectral range to improve the detection of methane plumes with high-resolution imaging spectrometers, 
# Atmos. Meas. Tech., 17, 1333–1346, https://doi.org/10.5194/amt-17-1333-2024, 2024.

import numpy as np
from numpy.typing import NDArray

from . import lut

# Import constants from lut module
FILE_LUT_GAS = lut.FILE_LUT_GAS
MAX_AMF = lut.MAX_AMF # Maximum AMF for PRISMA and EnMAP, used to avoid issues with the LUTs
read_luts = lut.read_luts

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


def AT_Combo_MF_2(mf_extended, mf_classic):
    std_swir = np.nanstd(mf_extended)
    std_2300 = np.nanstd(mf_classic)

    mf_combo = mf_extended.copy()

    mf_combo[mf_extended > mf_classic] = mf_classic[mf_extended > mf_classic]
    mf_combo[mf_extended <= mf_classic] = mf_combo[mf_extended <= mf_classic] * (
        std_2300 / std_swir
    )

    return mf_combo


def AT_MF_select_window_alt(img: NDArray, target_spectrum: NDArray) -> NDArray:
    """
    Vanilla matching filter implementation by column.

    Args:
        img (NDArray): image of shape (H, W, B) where H is the height, W is the width (number of columns) and B is the number of bands.
            Invalid values must be np.nan.
        target_spectrum (NDArray): target spectrum of shape (B,) or (W, B) where B is the number of bands and W is the number of columns.
            For EnMAP and EMIT it is a 1D array but for PRISMA it is 2D because there're slight changes in the wavelengths for each column
            (this seems to be because of the SMILE effect?)

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
        idxs_notnan = ~np.isnan(a)

        col_notnan = img[idxs_notnan, i] # (H', B)

        target_spectrum_i = target_spectrum[i] if len(target_spectrum.shape) == 2 else target_spectrum
        MF[idxs_notnan, i] = compute_mf_standard(col_notnan, target_spectrum_i)
        
    return MF


def compute_mf_standard(col_notnan: NDArray, target_spectrum: NDArray) -> NDArray:
    """
    Standard implementation of Roger et al. 2024's matching filter:
     mf = (x - µ)^T Σ^(-1) t / (t^T Σ^(-1) t)


    Args:
        col_notnan (NDArray): 2D array of shape (H', B) where H' is the number of rows with valid values and B is the number of bands.
        target_spectrum (NDArray): 1D array of shape (B,) representing the target spectrum.

    Returns:
        NDArray: 1D array of shape (H',) representing the matching filter response.
    """
    mu = np.nanmean(col_notnan, axis=0, keepdims=True) # (1, B)
    t = mu[0] * target_spectrum
    cov = np.cov(col_notnan, rowvar=False)
    X = col_notnan - mu  # X is the centered data matrix
    cov_inv = np.linalg.pinv(cov)

    cov_inv_reg_times_t = cov_inv @ t

    num = X @ cov_inv_reg_times_t
    den = t @ cov_inv_reg_times_t
    return num / den 


def compute_mf_reg1(col_notnan: NDArray, target_spectrum: NDArray, 
                    lambda_reg:float=0) -> NDArray:
    """
    Regularized implementation matching filter.

    This implementation is similar to the one implemented in mag1c. 

    Instead of inverting the covariance matrix directly, it uses SVD to compute the 
    regularized inverse of the covariance matrix Σ:
    
    Σ = U * diag(S) * V^T # Singular Value Decomposition of Σ
    (Σ + λI)^(-1) = V * diag(1 / (S + λ)) * V^T

    The final filter response is computed as:
     mf = (x - µ)^T (Σ + λI)^(-1) t

    Args:
        col_notnan (NDArray): 2D array of shape (H', B) where H' is the number of rows with valid values and B is the number of bands.
        target_spectrum (NDArray): 1D array of shape (B,) representing the target spectrum.
        lambda_reg (float, optional): Regularization parameter. Defaults to 0.

    Returns:
        NDArray: 1D array of shape (H',) representing the matching filter response.
    """
    mu = np.nanmean(col_notnan, axis=0, keepdims=True) # (1, B)
    t = mu[0] * target_spectrum
    cov = np.cov(col_notnan, rowvar=False)
    X = col_notnan - mu  # X is the centered data matrix
    U, S, Vt = np.linalg.svd(cov)

    S_inv_reg = np.diag(1. / (S + lambda_reg))  # Inverse with regularization

    # Step 4: Compute the inverse of the covariance matrix using the regularized SVD
    cov_inv_reg = Vt.T @ S_inv_reg @ U.T  # Regularized inverse via SVD

    cov_inv_reg_times_t = cov_inv_reg @ t

    # Step 5: Compute the numerator (x * A^T A + lambda I)^{-1} t
    num = X @ cov_inv_reg_times_t

    # Step 6: Compute the denominator (t^T (A^T A + lambda I)^{-1} t)
    den = t @ cov_inv_reg_times_t
    
    return num / den 

def compute_mf_reg2(col_notnan: NDArray, target_spectrum: NDArray, 
                    lambda_reg:float=0) -> NDArray:
    """
    Regularized implementation matching filter. This is matematically equivalent to compute_mf_reg1, but uses a different approach:

    Instead of inverting the covariance matrix directly, it uses SVD on the centered data matrix X 
    to compute the regularized inverse of the covariance matrix Σ. If X is the centered data matrix, then:
    X = U * diag(S) * V^T # Singular Value Decomposition of X
    (1/(M-1) * X^T X + λI)^(-1) = (M-1) V * diag(1 / (S^2 + λ)) * V^T

    The final filter response is computed as:
        mf = (x - µ)^T (1/(M-1) * X^T X + λI)^(-1) t

    Args:
        col_notnan (NDArray): 2D array of shape (H', B) where H' is the number of rows with valid values and B is the number of bands.
        target_spectrum (NDArray): 1D array of shape (B,) representing the target spectrum.
        lambda_reg (float, optional): Regularization parameter. Defaults to 0.

    Returns:
        NDArray: 1D array of shape (H',) representing the matching filter response.
    """
    
    mu = np.nanmean(col_notnan, axis=0, keepdims=True) # (1, B)
    t = mu[0] * target_spectrum
    
    X = col_notnan - mu  # X is the centered data matrix

    # Step 2: Perform SVD on the centered data matrix X
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    
    # Step 3: Add regularization to the squared singular values (S)
    S_squared_inv_reg = np.diag(1. / (S**2 + lambda_reg))  # Regularized inverse of S^2
    
    # Step 4: Compute the regularized inverse of X^T X using SVD
    # (Note: X^T X = V S^2 V^T, so we use the regularized inverse of S^2)
    M = X.shape[0]
    Sigma_inv_reg =  (M - 1) * Vt.T @ S_squared_inv_reg @ Vt  # Regularized inverse of 1/m X^T X

    # Step 5: (1/(M-1) * X^T X + lambda I)^{-1} t
    Sigma_inv_reg_times_t = Sigma_inv_reg @ t
    
    # Step 6: Compute the numerator (x * (1/(M-1) X^T X + lambda I)^{-1} t)
    num = X @ Sigma_inv_reg_times_t  # This step is equivalent to your numerator computation

    # Step 7: Compute the denominator (t^T (X^T X + lambda I)^{-1} t)
    den = t @ Sigma_inv_reg_times_t
    
    return num / den 
# Plume vetting paper: https://www.sciencedirect.com/science/article/pii/S0034425725002640
# Identification of false methane plumes for orbital imaging spectrometers: A case study with EMIT
# Chuchu Xiang, David R. Thompson, Robert O. Green, Jay E. Fahlen, Andrew K. Thorpe, Philip G. Brodrick, Red Willow Coleman, Amanda M. Lopez, Clayton D. Elder

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from georeader import griddata, rasterize, read
from georeader.geotensor import GeoTensor
from georeader.readers import enmap, prisma
from georeader.readers.emit import EMITImage
from numpy.typing import NDArray
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .emit.retrieval_upv_emit import load_target_spectrum_mf
from .plume_vetting_sfun import (
    calculate_dist,
    calculate_fit,
    calculate_magnitude,
    find_uniform_indices,
    get_radiance_ratio,
)
from .prismaenmap.retrieval_upv_prisma_enmap import target_spectrum_enmap, target_spectrum_prisma

logger = logging.getLogger(__name__)


# Empirical scaling from ppb-compatible LUT units to ppm·m for plume-vetting target signature.
# See design doc §3.1. Revisit after dedicated unit validation.
SCALE_TARGET_PPB_TO_PPMxM = 10.0


def how_many_pixels_does_polygon_occupy(polygon, ref_data:GeoTensor):
    rasterized = rasterize.rasterize_geometry_like(polygon, data_like=ref_data, value=1, fill=0, crs_geometry="EPSG:4326")
    number_of_pixels = np.sum(rasterized.values)
    return number_of_pixels, rasterized.values


def compute_other_plumes_exclusion_mask(
    vectors: list[Polygon], pol_idx_target: int, ref_data: GeoTensor
) -> GeoTensor:
    """Rasterize non-target polygons into a mask aligned to ``ref_data``."""
    other_polygons = [
        polygon_other
        for pol_idx_other, polygon_other in enumerate(vectors)
        if pol_idx_other != pol_idx_target
    ]
    if len(other_polygons) == 0:
        return GeoTensor(
            np.zeros_like(ref_data.values, dtype=bool),
            transform=ref_data.transform,
            crs=ref_data.crs,
            fill_value_default=False,
        )

    union_other_polygons = unary_union(other_polygons)
    return rasterize.rasterize_geometry_like(
        union_other_polygons,
        data_like=ref_data,
        value=1,
        fill=0,
        crs_geometry="EPSG:4326",
    )

def compute_masks(mf, mf_threshold=30, clouds_and_surface_water_mask=None, debug=False):
    ######### calculate bad pixel mask ################
    uni_row, uni_col = find_uniform_indices(mf)  # find rows and columns with two or fewer unique MF values
    rowcol_mask = np.full_like(clouds_and_surface_water_mask, False, dtype=bool)
    rowcol_mask[uni_row, :] = True
    rowcol_mask[:, uni_col] = True
    rowcol_mask[mf <= -9999] = True

    # This mask is used for target pixels. Combine the rowcol_mask with the clouds_and_surface_water_mask
    combined_mask = clouds_and_surface_water_mask | rowcol_mask

    # This mask is used for background pixels
    background_mask = combined_mask.copy()
    background_mask[(mf < -mf_threshold) | (mf > mf_threshold)] = True

    if debug:
        plt.imshow(combined_mask)
        plt.title("combined_mask")
        plt.show()
        plt.imshow(background_mask)
        plt.title("background_mask")
        plt.show()

    return combined_mask, background_mask

def coords_inside_plume_from_binary(thresholded_instance, debug=False):
    points_inside_plume = []
    minx, miny, maxx, maxy = 0, 0, thresholded_instance.shape[0], thresholded_instance.shape[1]
    for y in range(int(miny), int(maxy)):
        for x in range(int(minx), int(maxx)):
            if thresholded_instance[x,y]:
                points_inside_plume.append((y, x))

    if debug:
        debug_viz_data = np.zeros_like(thresholded_instance)
        for coords in points_inside_plume:
            y, x = coords
            debug_viz_data[x, y] = 1.
        plt.imshow(debug_viz_data)
        plt.show()

    return points_inside_plume


# Reference constant from the Xiang et al. prototype — kept for diagnostics.
# Do NOT use in the operational compute path; derive target_signature from the sensor LUT instead.
SCENE_TARGET_SIGNATURE_DEFAULT = np.asarray([9.65677880e-06, 9.44929334e-06, 9.04269344e-06, 8.85827101e-06, 8.98958035e-06, 8.49728372e-06, 8.62805714e-06, 8.80114715e-06, 8.33883158e-06, 8.16142819e-06, 7.73368404e-06, 7.39366692e-06, 7.35562982e-06, 7.04368029e-06, 6.11562144e-06, 6.53218762e-06, 6.63357218e-06, 5.78389142e-06, 5.84881855e-06, 5.47753266e-06, 4.79791781e-06, 1.13047456e-06, -2.35286791e-06, 4.59439270e-06, 4.56166689e-06, 3.33127292e-06, 1.54112465e-06, 3.26023549e-06, 3.48084766e-06, 2.07768389e-06, 1.34280623e-06, -1.43228457e-05, -5.30431391e-05, -1.43933273e-05, 1.97451749e-06, 2.04592925e-06, 1.06639434e-06, -5.74711749e-06, -1.36636143e-05, -8.58726232e-06, -5.89627356e-07, -2.22065616e-06, -3.23409978e-06, -2.62903649e-05, -2.16494465e-05, -5.92926997e-05, -2.94537249e-04, -3.05365354e-04, -6.61797152e-05, -3.59345933e-06, -9.58845533e-08, -6.98856370e-06, -2.20578091e-05, -4.63919870e-05, -9.98187735e-05, -1.06221661e-04, -1.34282159e-04, -8.69059866e-05, -3.41956800e-05,
                    -7.52936476e-06, -1.03710475e-06, -4.22974056e-05, -8.64636580e-05, -7.36054663e-05, -3.05240747e-04, -5.80721283e-04, -5.79022997e-04, -3.15508683e-03, -7.20142290e-03, -5.69085827e-03, -2.20352407e-03, -4.58517744e-04, -1.39371442e-04, -3.95658213e-05, -1.05974738e-05, -6.18653290e-06, -3.43563370e-05, -6.65568568e-05, -2.57255106e-04, -5.96432718e-04, -6.37708274e-04, -1.44363145e-03, -1.96407999e-03, -1.44413259e-03, -1.86983579e-03, -1.77950763e-03, -7.16888202e-04, -4.00458836e-04, -2.25835846e-04, -1.02706943e-04, -2.08844006e-05, -1.18141154e-07, 5.29685862e-07, -1.92552801e-07, -6.41317179e-06, -1.58942158e-04, -2.15882719e-04, -4.84205933e-04, -2.26379462e-03, -6.85842307e-03, -1.50126939e-02, -1.57474495e-02, -1.18301261e-02, -2.05601913e-02, -3.60834425e-02, -3.94273630e-02, -2.59847522e-02, -1.44271172e-02, -9.88838744e-03, -6.08667109e-03, -3.18633724e-03, -1.46129171e-03, -5.25151565e-04, -2.60101417e-04, -3.37112286e-04, -1.97430909e-04,
                    -8.00361556e-05, -1.10455109e-05, -1.31887203e-05, -4.19767046e-05, -3.32164105e-05, -2.49744551e-05, -7.15452259e-05, -3.87402691e-04, -2.02328613e-03, -7.70922092e-03, -1.23149560e-02, -2.02294852e-02, -2.76472263e-02, -2.87585064e-02, -2.83417192e-02, -3.16582578e-02, -2.93610347e-02, -3.23032725e-02, -3.81929431e-02, -3.90207197e-02, -3.29549597e-02, -3.48682336e-02, -3.12618589e-02, -2.05089329e-02, -1.49367305e-02, -1.07353593e-02, -5.97769992e-03, -4.02974442e-03, -2.81336234e-03, -2.08011794e-03, -2.54418498e-03, -2.72301685e-03, -1.72132735e-03, -9.38626716e-04, -5.69493323e-04, -4.56683372e-04, -3.41620643e-04, -2.17749254e-04, -1.29124440e-04, -9.29762234e-05, -6.92330156e-05, -5.67021771e-05, -6.96139050e-05, -6.38932674e-05, -4.11656578e-05, -3.67605277e-05, -7.57429864e-05, -2.71394567e-04, -1.01682139e-03, -3.75419836e-03, -1.26479782e-02, -3.75491237e-02, -8.38793133e-02, -1.13786173e-01, -1.16849649e-01, -8.59692033e-02, -1.38584499e-01,
                    -1.44295087e-01, -6.46457727e-02, -7.58295710e-02, -8.64002790e-02, -1.05684196e-01, -1.23224796e-01, -6.18103379e-02, -8.05863781e-02, -1.00805092e-01, -6.55649471e-02, -4.37739533e-02, -3.14234067e-02, -3.00320311e-02, -4.31862871e-02, -6.02494192e-02, -4.14973363e-02, -7.14254345e-02, -5.12582042e-02, -2.41301424e-02, -2.49192260e-02, -2.28203546e-02, -1.85601865e-02, -1.44834816e-02, -1.14326348e-02, -8.08636595e-03, -6.13045088e-03, -4.44505264e-03, -2.75363038e-03, -1.76782748e-03, -1.29662555e-03, -1.13509987e-03, -1.27273834e-03, -1.68181124e-03, -2.11492102e-03, -2.41104042e-03, -1.42739322e-03, -9.57903434e-04, -1.20352719e-03, -1.00580261e-03, -6.22048310e-04, -6.37473072e-04, -5.43213916e-04, -5.01025722e-04, -3.52837832e-04, -2.48386828e-04, -2.06771481e-04, -1.71915054e-04, -1.75783825e-04, -9.43188133e-05, -7.43165031e-05, -4.79766004e-05, -4.43891285e-05, -2.59220423e-05, -2.82430025e-05, -4.60777162e-05, -6.02059693e-05, -6.53891201e-05,
                    -9.07074440e-05, -1.41093437e-04, -2.96758124e-04, -7.87594072e-04, -2.17192119e-03, -5.58628668e-03, -1.28307233e-02, -2.48689824e-02, -4.44926900e-02, -6.73228909e-02, -7.92775824e-02, -7.57375261e-02, -5.13581699e-02, -3.93471604e-02, -2.78321330e-01, -3.55763275e-01, -9.97541568e-02, -1.58274724e-01, -2.33960939e-01, -3.49164500e-01, -4.32103636e-01, -5.05408477e-01, -5.59860848e-01, -6.24071005e-01, -5.76580213e-01, -5.90172912e-01, -6.55077760e-01, -8.45815529e-01, -9.15431133e-01, -5.23962589e-01, -7.43168727e-01, -7.78706093e-01, -5.71164846e-01, -8.73710217e-01, -1.14039687e+00, -1.03317436e+00, -6.33479317e-01, -8.39119315e-01, -1.10115544e+00, -5.98434457e-01, -5.94363617e-01, -5.53028991e-01, -4.19743280e-01, -2.49515472e-01, -2.23981385e-01, -2.46002450e-01, -1.57568885e-01, -1.20763738e-01, -1.00731546e-01, -7.19952086e-02, -4.84661609e-02, -3.50186769e-02, -2.38227621e-02, -2.20361438e-02, -1.95619810e-02])


def compute(
    radiance: NDArray,
    wavelengths: NDArray,
    cmf: GeoTensor,
    clouds_and_surface_water_mask: NDArray,
    target_signature: NDArray,
    polygons: list[Polygon],
    mf_threshold: float = 30,
    radius: int = 200,
    deg_poly: int = 10,
    num_pts: int = 40,
    min_polygon_size: int = 0,
    fit_wl_range: tuple[float, float] = (2100, 2440),
    random_seed: Optional[int] = None,
    logger=None,
) -> dict:
    """Compute plume-vetting scores for a list of candidate polygons.

    Args:
        radiance: TOA radiance array, shape (H, W, B). Masked pixels must be 0.
        wavelengths: Band centre wavelengths in nm, shape (B,).
        cmf: Matched-filter GeoTensor in ppm·m, shape (H, W). Must be on the same
            spatial grid as ``radiance``. Fill-value pixels are treated as 0.
        clouds_and_surface_water_mask: Boolean mask, shape (H, W). True = pixel
            must not be used as target or background. The caller is responsible
            for including, in addition to clouds and surface water, every pixel
            with non-finite or fill-value radiance/cmf. Concretely::

                clouds_and_surface_water_mask |= (
                    np.any(~np.isfinite(radiance), axis=-1)
                    | np.any(radiance == RADIANCE_FILL_VALUE, axis=-1)
                    | ~np.isfinite(cmf.values)
                    | (cmf.values == cmf.fill_value_default)
                )

            If this contract is broken, :func:`get_radiance_ratio` logs a
            descriptive error and returns ``None`` for the offending polygon,
            which is then skipped (other polygons are still scored). See the
            per-sensor wrappers (:func:`compute_emit`, :func:`compute_prisma`,
            :func:`compute_enmap`) for reference implementations.
        target_signature: Methane target spectrum in ppm·m on the same wavelength
            grid as ``wavelengths``. Either shape ``(B,)`` (same signature for every
            polygon) or shape ``(P, B)`` (one signature per polygon, where
            ``P = len(polygons)``).
        polygons: Candidate plume polygons in WGS84.
        mf_threshold: Half-width of the matched-filter range used to select background
            pixels (|MF| <= mf_threshold).
        radius: Search radius in pixels for background pixel matching.
        deg_poly: Degree of the polynomial continuum in the spectral fit.
        num_pts: Number of in-plume/background pixel pairs to use.
        min_polygon_size: Polygons with fewer pixels than this are skipped.
        fit_wl_range: (lo, hi) wavelength range in nm used for the spectral fit.
        random_seed: Seed for the local RandomState so runs are reproducible.
            None means non-deterministic (new seed each call).
        logger: Loguru-compatible logger. If None, uses the module-level loguru
            logger. Debug messages trace scene-level stats and per-polygon
            iteration; error messages signal mask-contract violations.

    Returns:
        dict keyed by polygon index. Each value is a dict with keys:
            ``D_norm``, ``alpha_con_len``, ``top_pairs``, ``coef``, ``fit_sig``,
            ``ratio``, ``wl_only_in_fit``.
        Polygons that are too small or produce no valid pairs are omitted.
    """
    rng = np.random.RandomState(random_seed)
    if logger is None:
        from loguru import logger as logger

    # Scene-level diagnostics — useful for cross-checking that the caller has
    # passed the same data the script-level debug observed.
    logger.debug(
        f"[compute] scene: radiance shape={radiance.shape}, "
        f"cmf shape={cmf.values.shape}, cmf.crs={cmf.crs}, "
        f"cmf.fill_value_default={cmf.fill_value_default}, "
        f"n_polygons={len(polygons)}, n_mask_True={int(clouds_and_surface_water_mask.sum())}"
    )
    logger.debug(
        f"[compute] radiance: min={float(radiance.min())}, max={float(radiance.max())}, "
        f"n_all_zero_pixels={int(np.all(radiance == 0, axis=-1).sum())}, "
        f"n_any_nan_pixels={int(np.any(~np.isfinite(radiance), axis=-1).sum())}"
    )

    # Prepare CMF: replace fill values with 0 so they don't affect masks/stats
    cmf = cmf.copy()
    cmf.values[cmf.values == cmf.fill_value_default] = 0
    cmf.fill_value_default = 0.0
    mf_values = cmf.values

    # Derive wavelength index arrays from the fit range
    wl = np.asarray(wavelengths)
    ind_fit = np.where((wl >= fit_wl_range[0]) & (wl <= fit_wl_range[1]))[0]
    ind_out = np.where((wl < fit_wl_range[0]) | (wl > fit_wl_range[1]))[0]
    wl_only_in_fit = wl[ind_fit]

    target_signature = np.asarray(target_signature)
    if target_signature.ndim == 1:
        if target_signature.shape[0] != wl.shape[0]:
            raise ValueError(
                f"target_signature has {target_signature.shape[0]} bands, expected {wl.shape[0]}"
            )
    elif target_signature.ndim == 2:
        if target_signature.shape[0] != len(polygons):
            raise ValueError(
                f"target_signature has {target_signature.shape[0]} rows, expected {len(polygons)}"
            )
        if target_signature.shape[1] != wl.shape[0]:
            raise ValueError(
                f"target_signature has {target_signature.shape[1]} bands, expected {wl.shape[0]}"
            )
    else:
        raise ValueError(
            f"target_signature must be 1-D or 2-D, got shape {target_signature.shape}"
        )

    scores_per_polygon = {}
    for pol_idx, polygon in enumerate(polygons):
        if target_signature.ndim == 1:
            sig = target_signature[ind_fit]
        else:
            sig = target_signature[pol_idx, ind_fit]

        number_of_pixels, plume_mask = how_many_pixels_does_polygon_occupy(polygon, cmf)
        if number_of_pixels <= min_polygon_size:
            logger.debug(
                f"[compute pol_idx={pol_idx}] skip: only {int(number_of_pixels)} "
                f"in-plume pixels (<= min_polygon_size={min_polygon_size})"
            )
            continue

        # Per-polygon diagnostics. radiance[plume_mask_bool] is the polygon's
        # post-fill-substitution radiance; all-zero rows here are pixels that
        # were originally fill in at least one band (caught by the contract
        # mask). cmf has already had fill substituted to 0 above.
        plume_mask_bool = plume_mask.astype(bool)
        rdn_in_poly = radiance[plume_mask_bool]
        cmf_in_poly = mf_values[plume_mask_bool]
        n_all_zero_in_poly = int(np.all(rdn_in_poly == 0, axis=-1).sum())
        logger.debug(
            f"[compute pol_idx={pol_idx}] in_plume_pixels={int(number_of_pixels)}, "
            f"rdn mean={float(rdn_in_poly.mean()):.4f} (across all bands), "
            f"cmf mean={float(cmf_in_poly.mean()):.4f}, "
            f"cmf max={float(cmf_in_poly.max()):.4f}, "
            f"n_all_zero_rdn_in_poly={n_all_zero_in_poly}, "
            f"polygon_bounds={polygon.bounds}"
        )

        combined_mask, background_mask = compute_masks(
            mf_values, mf_threshold,
            clouds_and_surface_water_mask=clouds_and_surface_water_mask,
        )
        exclusion_mask = compute_other_plumes_exclusion_mask(polygons, pol_idx, cmf)
        background_mask = background_mask | exclusion_mask.values.astype(bool)
        orig_points_inside_plume = coords_inside_plume_from_binary(plume_mask, False)

        results = get_radiance_ratio(
            wl, radius, num_pts, radiance, mf_values,
            None,  # plume_coord (unused when ite=0)
            orig_points_inside_plume, ind_out,
            0,  # ite=0: use original plume position
            0,  # ii: loop counter (CSV saving, not used here)
            combined_mask, background_mask, plume_mask,
            1,  # dist_opt=1: L1-normalised similarity
            rng=rng,
            logger=logger,
        )
        if results is None:
            logger.debug(
                f"[compute pol_idx={pol_idx}] skipped (get_radiance_ratio "
                f"returned None — see error logs above for the reason)"
            )
            continue

        _contour_coord, _similarity, ratio, _top_ind, _top_mf, _avg_top_mf, _avg_in_plume_mf, top_pairs = results

        ratio_fit = np.array(ratio)[ind_fit]
        coef, fit_sig = calculate_fit(deg_poly, wl_only_in_fit, sig, ratio_fit)
        coef[0] *= 1e5  # scale estimated concentration length
        alpha_con_len = coef[0]

        polyn = np.polyval(coef[1:], wl_only_in_fit)
        ratio_p = ratio_fit / polyn
        sig_p = fit_sig / polyn

        dist_t = calculate_dist(ratio_p, sig_p, 0)
        mag_t = calculate_magnitude(ratio_p, 1)
        dist_t /= mag_t

        scores_per_polygon[pol_idx] = {
            "D_norm": dist_t,
            "alpha_con_len": alpha_con_len,
            "top_pairs": top_pairs,
            "coef": coef,
            "fit_sig": fit_sig,
            "ratio": ratio_fit,
            "wl_only_in_fit": wl_only_in_fit,
        }

    return scores_per_polygon


def plot_vetting(result: dict, cmf_values: NDArray) -> None:
    """Plot the two diagnostic figures for a single polygon's vetting result.

    Args:
        result: One polygon's entry from the dict returned by :func:`compute`.
        cmf_values: 2-D matched-filter array (H, W) for the scene, used as the
            background image in the pixel-pair figure.
    """
    top_pairs = result["top_pairs"]
    coef = result["coef"]
    fit_sig = result["fit_sig"]
    ratio = result["ratio"]
    wl = result["wl_only_in_fit"]

    # Fig 1: target (in-plume) and background pixel locations on the CMF image
    points_A = np.array([[coord_a[1], coord_a[0]] for coord_a, _, _ in top_pairs])
    points_B = np.array([[coord_b[1], coord_b[0]] for _, coord_b, _ in top_pairs])

    all_x = np.concatenate([points_A[:, 0], points_B[:, 0]])
    all_y = np.concatenate([points_A[:, 1], points_B[:, 1]])
    pad = max(10, int(0.1 * max(all_x.max() - all_x.min(), all_y.max() - all_y.min())))
    xlim = (all_x.min() - pad, all_x.max() + pad)
    ylim = (all_y.max() + pad, all_y.min() - pad)  # imshow y-axis: larger row = lower

    cmf_display = cmf_values / 8000.0
    cmf_display = np.where(cmf_values <= 0, 0, cmf_display)
    fig, (ax_plain, ax_scatter) = plt.subplots(1, 2, sharey=True)
    fig.suptitle(f"D_norm={result['D_norm']:.3f}  α_con_len={result['alpha_con_len']:.3f}")
    ax_plain.imshow(cmf_display, vmin=0, vmax=0.3)
    ax_plain.set_xlim(xlim)
    ax_plain.set_ylim(ylim)
    ax_scatter.imshow(cmf_display, vmin=0, vmax=0.3)
    ax_scatter.scatter(points_A[:, 0], points_A[:, 1], s=12, color="orange", label="in-plume")
    ax_scatter.scatter(points_B[:, 0], points_B[:, 1], s=12, color="red", label="out-of-plume")
    ax_scatter.set_xlim(xlim)
    ax_scatter.legend()
    plt.show()

    # Fig 2: spectral fit — measurement vs model, with and without continuum
    yy = np.polyval(coef[1:], wl)
    y1 = ratio.copy()
    y2 = fit_sig.copy()

    fig, axes = plt.subplots(2, 1, constrained_layout=True, squeeze=False)
    axes = axes.flatten()
    ax_upper = axes[0]
    ax_upper.plot(wl, yy, label="Continuum function", color="green", linestyle="--", alpha=0.5)
    line1, = ax_upper.plot(wl, y1, label="Measurement")
    line2, = ax_upper.plot(wl, y2, label="Model")

    ax_lower = axes[1]
    ax_lower.plot(wl, y1 / yy, label="Measurement / continuum", color=line1.get_color(), linestyle="--")
    ax_lower.plot(wl, y2 / yy, label="Model / continuum", color=line2.get_color(), linestyle="--")
    ax_lower.set_xlabel("Wavelength (nm)")

    handles_u, labels_u = ax_upper.get_legend_handles_labels()
    handles_l, labels_l = ax_lower.get_legend_handles_labels()
    plt.legend(handles_u + handles_l, labels_u + labels_l, fontsize=8)
    plt.show()


EMIT_RADIANCE_FILL_VALUE = -9999
PRISMA_RADIANCE_FILL_VALUE = -20


def compute_emit(
    emit_image: EMITImage,
    cmf: GeoTensor,
    polygons: list[Polygon],
    target_signature: Optional[NDArray] = None,
    num_pts: int = 40,
    min_polygon_size: int = 0,
    mf_threshold: float = 30,
    radius: int = 200,
    deg_poly: int = 10,
    fit_wl_range: tuple[float, float] = (2100, 2440),
    random_seed: Optional[int] = None,
    use_l2a_mask: bool = False,
    logger=None,
) -> dict:
    """Plume vetting for an EMIT scene — convenience wrapper around :func:`compute`.

    Extracts radiance, wavelengths, and the cloud/surface-water mask directly from
    ``emit_image``, builds the LUT-derived target signature via
    :func:`~marshsi.emit.retrieval_upv_emit.load_target_spectrum_mf`, and delegates
    to :func:`compute`.

    The mask passed to :func:`compute` is the union of (a) non-finite / fill-value
    pixels in the radiance, (b) non-finite / fill-value pixels in the CMF, and
    optionally (c) the EMIT L2A cloud + surface-water mask. See ``use_l2a_mask``.

    Args:
        emit_image: Loaded EMIT scene (provides radiance, wavelengths, mask).
        cmf: Matched-filter GeoTensor in ppm·m. Reprojected to the radiance grid
            automatically if extents differ.
        polygons: Candidate plume polygons in WGS84.
        target_signature: Override the LUT-derived target spectrum (shape ``(B,)``,
            in ppm·m). If ``None`` (default), uses
            ``load_target_spectrum_mf(emit_image) * SCALE_TARGET_PPB_TO_PPMxM``.
        num_pts: Number of in-plume / background pixel pairs.
        min_polygon_size: Polygons with fewer pixels are skipped.
        mf_threshold: Background pixel MF half-width threshold.
        radius: Background search radius in pixels.
        deg_poly: Polynomial continuum degree for the spectral fit.
        fit_wl_range: (lo, hi) wavelength window in nm for the spectral fit.
        random_seed: Seed for reproducible pixel-pair selection.
        use_l2a_mask: When True, OR the EMIT L2A cloud + surface-water mask
            (bands 0-2) into the mask passed to :func:`compute`. Defaults to
            False because the EMIT L2A cloud mask is known to be unreliable
            (frequently flags real plume pixels). When False, only the
            radiance/CMF invalidity mask is applied.
        logger: Loguru-compatible logger. Passed through to :func:`compute`.
            If None, uses the module-level loguru logger.

    Returns:
        Same dict as :func:`compute` — keyed by polygon index, each value contains
        ``D_norm``, ``alpha_con_len``, and the diagnostic arrays for :func:`plot_vetting`.
    """

    if logger is None:
        logger = logging.getLogger(__name__)
    # Build target signature from LUT if not supplied
    if target_signature is None:
        target_signature = load_target_spectrum_mf(emit_image) * SCALE_TARGET_PPB_TO_PPMxM

    # Load radiance and reshape to (H, W, B). Keep the raw array around so we
    # can build the invalidity mask from it before replacing fills with 0.
    #
    # NOTE: we deliberately bypass emit_image.load() (which wraps
    # load_raw + georreference) and call the two stages explicitly so we can
    # log the raw sensor-frame radiance before GLT orthorectification. This is
    # diagnostic: it tells us whether the data is already zero/fill at the
    # netCDF level (upstream/state bug) or whether the GLT step is producing
    # zeros (orthorectification bug).
    # load_raw() default transpose=True returns (C, H, W), which is what
    # emit_image.georreference() expects. Stats reduced along axis 0 (bands).
    rdn_sensor_arr = np.asarray(emit_image.load_raw())  # (C, H, W)
    # Diagnostic format matches marshsi.emit.retrieval_upv_emit.AT_MF_total_EMIT
    # and marshsi.emit.mag1c_emit so the three call sites can be correlated.
    import os as _os
    _p = emit_image.filename
    logger.debug(
        f"[compute_emit.load_raw] file={_p} "
        f"size={_os.path.getsize(_p)} mtime={_os.path.getmtime(_p):.1f} "
        f"shape={rdn_sensor_arr.shape}, dtype={rdn_sensor_arr.dtype}, "
        f"min={float(rdn_sensor_arr.min())}, max={float(rdn_sensor_arr.max())}, "
        f"n_neg9999={int(np.sum(rdn_sensor_arr == EMIT_RADIANCE_FILL_VALUE))}, "
        f"n_nan={int(np.sum(~np.isfinite(rdn_sensor_arr)))}, "
        f"n_all_zero_pixels={int(np.sum(np.all(rdn_sensor_arr == 0, axis=0)))}, "
        f"n_pixels_total={int(rdn_sensor_arr.shape[1] * rdn_sensor_arr.shape[2])}"
    )

    data = emit_image.georreference(
        rdn_sensor_arr, fill_value_default=EMIT_RADIANCE_FILL_VALUE
    )
    rdn_raw = np.transpose(data.values, (1, 2, 0))   # (B, H, W) → (H, W, B)

    logger.debug(
        f"[compute_emit] emit_image.crs={getattr(emit_image, 'crs', None)}, "
        f"data.crs={data.crs}, data.transform={data.transform}, "
        f"rdn_raw shape={rdn_raw.shape}, dtype={rdn_raw.dtype}, "
        f"rdn_raw min={float(rdn_raw.min())}, max={float(rdn_raw.max())}, "
        f"n_neg9999={int(np.sum(rdn_raw == EMIT_RADIANCE_FILL_VALUE))}, "
        f"n_nan={int(np.sum(~np.isfinite(rdn_raw)))}, "
        f"n_all_zero_pixels={int(np.sum(np.all(rdn_raw == 0, axis=-1)))}, "
        f"data.fill_value_default={data.fill_value_default}"
    )
    logger.debug(
        f"[compute_emit] cmf.crs={cmf.crs}, cmf.transform={cmf.transform}, "
        f"cmf shape={cmf.values.shape}, cmf.fill_value_default={cmf.fill_value_default}, "
        f"cmf min={float(cmf.values.min())}, cmf max={float(cmf.values.max())}, "
        f"n_cmf_fill={int(np.sum(cmf.values == cmf.fill_value_default))}, "
        f"same_extent={cmf.same_extent(data)}"
    )

    # Reproject CMF to the radiance grid if extents differ
    if not cmf.same_extent(data):
        logger.debug("[compute_emit] cmf differs in extent → reprojecting via read_reproject_like")
        cmf = read.read_reproject_like(cmf, data)
        logger.debug(
            f"[compute_emit] cmf after reproject: shape={cmf.values.shape}, "
            f"fill_value_default={cmf.fill_value_default}, "
            f"min={float(cmf.values.min())}, max={float(cmf.values.max())}, "
            f"n_fill={int(np.sum(cmf.values == cmf.fill_value_default))}"
        )

    # Build the mask required by compute()'s contract: pixels with non-finite
    # or fill-value radiance, plus pixels with non-finite or fill-value CMF.
    # Optionally OR in the L2A cloud + surface-water mask (off by default).
    rdn_invalid = (
        np.any(~np.isfinite(rdn_raw), axis=-1)
        | np.any(rdn_raw == EMIT_RADIANCE_FILL_VALUE, axis=-1)
    )
    cmf_invalid = ~np.isfinite(cmf.values) | (cmf.values == cmf.fill_value_default)
    mask = rdn_invalid | cmf_invalid

    if use_l2a_mask:
        mask_raw = np.array(emit_image.nc_ds_l2amask["mask"])
        l2a_mask = np.sum(mask_raw[..., :3], axis=-1) > 0
        l2a_mask_geo = emit_image.georreference(l2a_mask, fill_value_default=True)
        mask = mask | l2a_mask_geo.values

    logger.debug(
        f"[compute_emit] mask built: rdn_invalid={int(rdn_invalid.sum())}, "
        f"cmf_invalid={int(cmf_invalid.sum())}, total mask True={int(mask.sum())} "
        f"of {mask.size} pixels ({100.0 * mask.sum() / mask.size:.2f}%)"
    )

    rdn = np.where(rdn_raw == EMIT_RADIANCE_FILL_VALUE, 0, rdn_raw)

    wavelengths = np.asarray(emit_image.wavelengths)

    return compute(
        radiance=rdn,
        wavelengths=wavelengths,
        cmf=cmf,
        clouds_and_surface_water_mask=mask,
        target_signature=target_signature,
        polygons=polygons,
        mf_threshold=mf_threshold,
        radius=radius,
        deg_poly=deg_poly,
        num_pts=num_pts,
        min_polygon_size=min_polygon_size,
        fit_wl_range=fit_wl_range,
        random_seed=random_seed,
        logger=logger,
    )


def compute_prisma(
    pi: prisma.PRISMA,
    cmf: GeoTensor,
    polygons: list[Polygon],
    target_signature: Optional[NDArray] = None,
    num_pts: int = 40,
    min_polygon_size: int = 0,
    mf_threshold: float = 30,
    radius: int = 200,
    deg_poly: int = 10,
    fit_wl_range: tuple[float, float] = (2100, 2440),
    random_seed: Optional[int] = None,
    logger=None,
) -> dict:
    """Plume vetting for a PRISMA scene — convenience wrapper around :func:`compute`.

    Uses PRISMA SWIR radiance and delegates to :func:`compute`. The SMILE effect is
    **not** accounted for: both wavelengths and the default target signature are
    averaged across all PRISMA columns, producing a single scene-wide spectrum.

    Args:
        pi: Loaded PRISMA scene.
        cmf: Matched-filter GeoTensor for the scene.
        polygons: Candidate plume polygons in WGS84.
        target_signature: Override the default target spectrum. Accepted shapes:
            ``(B,)`` (one signature for all polygons) or ``(P, B)`` (one per polygon).
            If ``None``, builds a scene-average signature with
            ``target_spectrum_prisma(pi, swir_flag=True)`` averaged over columns and
            scaled by ``SCALE_TARGET_PPB_TO_PPMxM``.
        num_pts: Number of in-plume / background pixel pairs.
        min_polygon_size: Polygons with fewer pixels are skipped.
        mf_threshold: Background pixel MF half-width threshold.
        radius: Background search radius in pixels.
        deg_poly: Polynomial continuum degree for the spectral fit.
        fit_wl_range: (lo, hi) wavelength window in nm for the spectral fit.
        random_seed: Seed for reproducible pixel-pair selection.

    Returns:
        Same dict as :func:`compute`.
    """

    # PRISMA raw SWIR is loaded in (column, row, band); transpose to (row, col, band).
    rdn_raw = np.transpose(np.asarray(pi.load_raw(swir_flag=True)), (1, 0, 2))
    rdn_geo = griddata.read_to_crs(
        rdn_raw.astype(np.float64),
        lons=pi.lons,
        lats=pi.lats,
        resolution_dst=30,
        fill_value_default=PRISMA_RADIANCE_FILL_VALUE,
        dst_crs=cmf.crs,
    )

    if not cmf.same_extent(rdn_geo):
        cmf = read.read_reproject_like(cmf, rdn_geo)

    # Build the mask required by compute()'s contract: pixels with non-finite
    # or fill-value radiance, plus pixels with non-finite or fill-value CMF.
    # rdn_geo.values has shape (B, H, W); reduce along the band axis.
    rdn_invalid = np.any(~np.isfinite(rdn_geo.values), axis=0) | np.any(
        rdn_geo.values <= PRISMA_RADIANCE_FILL_VALUE, axis=0
    )
    cmf_invalid = ~np.isfinite(cmf.values) | (cmf.values == cmf.fill_value_default)
    mask = rdn_invalid | cmf_invalid

    # Transpose radiance to (H, W, B) and replace fill / non-finite values with 0
    rdn_vals = np.transpose(rdn_geo.values, (1, 2, 0))
    rdn_vals = np.where(np.isfinite(rdn_vals), rdn_vals, 0.0)
    rdn_vals = np.where(rdn_vals <= PRISMA_RADIANCE_FILL_VALUE, 0.0, rdn_vals)

    wavelengths = np.mean(np.asarray(pi.wavelength_swir), axis=0)

    if target_signature is None:
        target_signature = (
            target_spectrum_prisma(pi, swir_flag=True) * SCALE_TARGET_PPB_TO_PPMxM
        )
        # average the target signature.
        target_signature = np.mean(target_signature, axis=0)  # (M, B) → (B,)

    return compute(
        radiance=rdn_vals,
        wavelengths=wavelengths,
        cmf=cmf,
        clouds_and_surface_water_mask=mask,
        target_signature=target_signature,
        polygons=polygons,
        mf_threshold=mf_threshold,
        radius=radius,
        deg_poly=deg_poly,
        num_pts=num_pts,
        min_polygon_size=min_polygon_size,
        fit_wl_range=fit_wl_range,
        random_seed=random_seed,
        logger=logger,
    )


def compute_enmap(
    enmapi: enmap.EnMAP,
    cmf: GeoTensor,
    polygons: list[Polygon],
    target_signature: Optional[NDArray] = None,
    num_pts: int = 40,
    min_polygon_size: int = 0,
    mf_threshold: float = 30,
    radius: int = 200,
    deg_poly: int = 10,
    fit_wl_range: tuple[float, float] = (2100, 2440),
    random_seed: Optional[int] = None,
    logger=None,
) -> dict:
    """Plume vetting for an EnMAP scene — convenience wrapper around :func:`compute`.

    Extracts EnMAP SWIR radiance/wavelengths, builds the LUT-derived target
    signature via :func:`~marshsi.prismaenmap.retrieval_upv_prisma_enmap.target_spectrum_enmap`,
    and delegates to :func:`compute`.

    Args:
        enmapi: Loaded EnMAP scene.
        cmf: Matched-filter GeoTensor in ppm·m. Reprojected to the radiance grid
            automatically if extents differ.
        polygons: Candidate plume polygons in WGS84.
        target_signature: Override the LUT-derived target spectrum (shape ``(B,)``,
            in ppm·m). If ``None`` (default), uses
            ``target_spectrum_enmap(enmapi) * SCALE_TARGET_PPB_TO_PPMxM``.
        num_pts: Number of in-plume / background pixel pairs.
        min_polygon_size: Polygons with fewer pixels are skipped.
        mf_threshold: Background pixel MF half-width threshold.
        radius: Background search radius in pixels.
        deg_poly: Polynomial continuum degree for the spectral fit.
        fit_wl_range: (lo, hi) wavelength window in nm for the spectral fit.
        random_seed: Seed for reproducible pixel-pair selection.

    Returns:
        Same dict as :func:`compute`.
    """

    if target_signature is None:
        target_signature = target_spectrum_enmap(enmapi) * SCALE_TARGET_PPB_TO_PPMxM

    data_swir = enmapi.load_product("SPECTRAL_IMAGE_SWIR")
    fill_val = data_swir.fill_value_default
    # apply RPC EnMAP (needed to have colocated data with the plume).
    data_swir = read.read_rpcs(
            data_swir.values.astype(np.float32),
            rpcs=enmapi.rpcs_swir,
            dst_crs=cmf.crs,
            resolution_dst_crs=30,
            fill_value_default=fill_val,
        )

    rdn = np.transpose(data_swir.values, (1, 2, 0)).astype(np.float64)

    if not cmf.same_extent(data_swir):
        cmf = read.read_reproject_like(cmf, data_swir)

    # Build the mask required by compute()'s contract: pixels with non-finite
    # or fill-value radiance, plus pixels with non-finite or fill-value CMF.
    rdn_invalid = np.any(~np.isfinite(rdn), axis=-1) | np.any(rdn == fill_val, axis=-1)
    cmf_invalid = ~np.isfinite(cmf.values) | (cmf.values == cmf.fill_value_default)
    mask = rdn_invalid | cmf_invalid

    rdn[rdn == fill_val] = 0.0
    rdn = np.where(~np.isfinite(rdn), 0.0, rdn)

    wavelengths = np.asarray(enmapi.wl_center["swir"])

    return compute(
        radiance=rdn,
        wavelengths=wavelengths,
        cmf=cmf,
        clouds_and_surface_water_mask=mask,
        target_signature=target_signature,
        polygons=polygons,
        mf_threshold=mf_threshold,
        radius=radius,
        deg_poly=deg_poly,
        num_pts=num_pts,
        min_polygon_size=min_polygon_size,
        fit_wl_range=fit_wl_range,
        random_seed=random_seed,
        logger=logger,
    )

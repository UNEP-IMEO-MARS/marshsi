"""Tests for marshsi.plume_vetting — contract enforcement and mask augmentation.

These tests cover the two pieces of behaviour added to defend against silently
producing all-NaN ``similarity_matrix`` inputs to ``scipy.optimize.linear_sum_assignment``:

1. ``plume_vetting_sfun._check_radiance_rows_valid`` raises a descriptive
   ``ValueError`` when the caller's ``clouds_and_surface_water_mask`` lets a
   non-finite or all-zero radiance row through.
2. The per-sensor wrappers (``compute_emit`` / ``compute_prisma`` / ``compute_enmap``)
   build the mask required by ``compute``'s contract — non-finite and fill-value
   pixels in both the radiance and the CMF are flagged automatically.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest
from affine import Affine
from georeader.geotensor import GeoTensor
from shapely.geometry import Polygon

import marshsi.plume_vetting as pv
from marshsi.plume_vetting_sfun import _check_radiance_rows_valid

# ── helper unit tests ───────────────────────────────────────────────────


class TestCheckRadianceRowsValid:
    """The helper returns a bool and logs an error; it no longer raises.

    A False return tells ``get_radiance_ratio`` to bail out (return None) so
    ``compute()`` can skip the offending polygon and continue with the rest —
    we'd rather diagnose which polygon broke the contract than abort the tile.
    """

    def test_passes_on_clean_array(self, caplog):
        arr = np.ones((5, 10))
        mock_logger = MagicMock()
        assert _check_radiance_rows_valid(arr, "A_data", logger=mock_logger) is True
        mock_logger.error.assert_not_called()

    def test_returns_false_on_nan_row(self):
        arr = np.ones((5, 10))
        arr[2, 3] = np.nan
        mock_logger = MagicMock()
        assert _check_radiance_rows_valid(arr, "A_data (target)", logger=mock_logger) is False
        mock_logger.error.assert_called_once()
        msg = mock_logger.error.call_args[0][0]
        assert "A_data (target)" in msg
        assert "1 non-finite row" in msg
        assert "0 all-zero row" in msg
        assert "clouds_and_surface_water_mask" in msg

    def test_returns_false_on_inf_row(self):
        arr = np.ones((3, 4))
        arr[1, 0] = np.inf
        mock_logger = MagicMock()
        assert _check_radiance_rows_valid(arr, "B_data (background)", logger=mock_logger) is False
        assert "non-finite row" in mock_logger.error.call_args[0][0]

    def test_returns_false_on_all_zero_row(self):
        arr = np.ones((4, 6))
        arr[0, :] = 0.0
        mock_logger = MagicMock()
        assert _check_radiance_rows_valid(arr, "B_data (background)", logger=mock_logger) is False
        msg = mock_logger.error.call_args[0][0]
        assert "1 all-zero row" in msg
        assert "0 non-finite row" in msg

    def test_counts_combined_non_finite_and_zero_rows(self):
        arr = np.ones((6, 5))
        arr[0, :] = 0.0
        arr[1, :] = 0.0
        arr[3, 2] = np.nan
        mock_logger = MagicMock()
        assert _check_radiance_rows_valid(arr, "X", logger=mock_logger) is False
        msg = mock_logger.error.call_args[0][0]
        assert "1 non-finite row" in msg
        assert "2 all-zero row" in msg
        assert "6 total" in msg

    def test_error_message_points_to_contract(self):
        arr = np.zeros((1, 3))
        mock_logger = MagicMock()
        _check_radiance_rows_valid(arr, "A_data", logger=mock_logger)
        assert "marshsi.plume_vetting.compute" in mock_logger.error.call_args[0][0]

    def test_default_logger_used_when_none(self):
        """Passing logger=None falls back to the module-level loguru logger
        (no exception raised, just a default-logged error)."""
        arr = np.zeros((1, 3))
        assert _check_radiance_rows_valid(arr, "A_data") is False


# ── compute() contract enforcement via a synthetic scene ───────────────


def _build_synthetic_scene(
    h: int = 40,
    w: int = 40,
    n_bands: int = 50,
    plume_origin: tuple[int, int] = (15, 15),
    plume_size: int = 6,
    rng_seed: int = 0,
):
    """Build a tiny in-memory scene: radiance, wavelengths, CMF, target signature, polygon.

    The scene uses an EPSG:32610 grid at (origin_x, origin_y) = (500000, 4000000)
    with 30 m resolution. Returns a dict with everything ``compute()`` needs.
    """
    rng = np.random.RandomState(rng_seed)

    # Wavelengths spanning 400-2500 nm so the fit window (2100-2440) is covered
    wavelengths = np.linspace(400.0, 2500.0, n_bands)

    # Radiance: small positive baseline + per-pixel structure so rows are non-degenerate
    radiance = 1.0 + 0.05 * rng.rand(h, w, n_bands)

    # CMF: high MF inside the plume polygon, low-level noise elsewhere.
    # Noise is needed because find_uniform_indices() in plume_vetting_sfun
    # flags any row/column with ≤2 unique values as "uniform" and masks it.
    transform = Affine(30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0)
    cmf_values = (5.0 * rng.rand(h, w)).astype(np.float32)  # |MF| < mf_threshold=30
    py, px = plume_origin
    cmf_values[py:py + plume_size, px:px + plume_size] = 500.0  # well above mf_threshold
    cmf = GeoTensor(
        cmf_values,
        transform=transform,
        crs="EPSG:32610",
        fill_value_default=-9999.0,
    )

    # Polygon in WGS84 covering the high-MF region.
    # Compute lon/lat for the plume corners. Easier to construct directly from
    # transform: pixel (col, row) → (x, y) = transform * (col, row).
    x0, y0 = transform * (px, py)
    x1, y1 = transform * (px + plume_size, py + plume_size)
    # Convert UTM zone-10 metres → lon/lat via pyproj
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
    lon0, lat0 = t.transform(x0, y0)
    lon1, lat1 = t.transform(x1, y1)
    polygon = Polygon([
        (min(lon0, lon1), min(lat0, lat1)),
        (min(lon0, lon1), max(lat0, lat1)),
        (max(lon0, lon1), max(lat0, lat1)),
        (max(lon0, lon1), min(lat0, lat1)),
    ])

    # Target signature: gentle dip across the fit window
    target_signature = np.ones(n_bands)
    fit_mask = (wavelengths >= 2100) & (wavelengths <= 2440)
    target_signature[fit_mask] = 0.5

    # Empty mask (caller asserts everything is valid)
    clouds_and_surface_water_mask = np.zeros((h, w), dtype=bool)

    return {
        "radiance": radiance,
        "wavelengths": wavelengths,
        "cmf": cmf,
        "clouds_and_surface_water_mask": clouds_and_surface_water_mask,
        "target_signature": target_signature,
        "polygons": [polygon],
    }


class TestComputeContractEnforcement:
    def test_unmask_fill_pixel_inside_polygon_skips_polygon_and_logs(self):
        scene = _build_synthetic_scene()
        # Plant a zero-radiance pixel (simulating un-substituted fill) inside the
        # polygon WITHOUT flagging it in the mask. compute() should log an
        # error from _check_radiance_rows_valid and return an empty dict
        # (this polygon was the only one, and it gets skipped).
        scene["radiance"][16, 16, :] = 0.0  # inside plume polygon

        mock_logger = MagicMock()
        result = pv.compute(
            radius=8,
            num_pts=4,
            min_polygon_size=0,
            random_seed=0,
            logger=mock_logger,
            **scene,
        )
        assert result == {}
        # The validator should have logged the contract violation
        assert mock_logger.error.call_count >= 1
        error_msgs = " ".join(c[0][0] for c in mock_logger.error.call_args_list)
        assert "A_data" in error_msgs or "B_data" in error_msgs
        assert "all-zero row" in error_msgs or "non-finite row" in error_msgs

    def test_unmask_nan_pixel_inside_polygon_skips_polygon_and_logs(self):
        scene = _build_synthetic_scene()
        scene["radiance"][16, 16, 5] = np.nan  # one NaN entry inside plume polygon

        mock_logger = MagicMock()
        result = pv.compute(
            radius=8,
            num_pts=4,
            min_polygon_size=0,
            random_seed=0,
            logger=mock_logger,
            **scene,
        )
        assert result == {}
        error_msgs = " ".join(c[0][0] for c in mock_logger.error.call_args_list)
        assert "non-finite row" in error_msgs

    def test_masking_the_bad_pixel_lets_compute_run(self, recwarn):
        """When the caller correctly flags the bad pixel, the polygon is skipped
        cleanly inside get_radiance_ratio (returns None) and compute() returns
        a dict — no ValueError."""
        import warnings as _warnings

        from scipy.optimize import OptimizeWarning

        scene = _build_synthetic_scene()
        scene["radiance"][16, 16, :] = 0.0
        scene["clouds_and_surface_water_mask"][16, 16] = True

        # Synthetic radiance doesn't match the target signature well enough for
        # curve_fit to estimate a covariance — that's irrelevant here. We only
        # care that compute() doesn't raise a ValueError on the contract path.
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", OptimizeWarning)
            result = pv.compute(
                radius=8,
                num_pts=4,
                min_polygon_size=0,
                random_seed=0,
                **scene,
            )
        assert isinstance(result, dict)


# ── per-sensor mask augmentation: compute_emit ─────────────────────────


def _mock_emit_image(rdn_raw: np.ndarray, wavelengths: np.ndarray, transform: Affine,
                     crs: str, l2a_mask: np.ndarray | None = None) -> MagicMock:
    """Build a MagicMock with the surface area of an EMITImage that compute_emit uses."""
    # compute_emit calls emit_image.load(as_reflectance=False), then transposes
    # data.values from (B, H, W) → (H, W, B).
    data_values_bhw = np.transpose(rdn_raw, (2, 0, 1))
    data_geotensor = GeoTensor(
        data_values_bhw,
        transform=transform,
        crs=crs,
        fill_value_default=pv.EMIT_RADIANCE_FILL_VALUE,
    )

    emit_image = MagicMock()
    emit_image.load.return_value = data_geotensor
    emit_image.wavelengths = wavelengths
    if l2a_mask is not None:
        # nc_ds_l2amask["mask"] is indexed mask_raw[..., :3] — give it 3 bands
        # of the same boolean mask for simplicity.
        emit_image.nc_ds_l2amask = {
            "mask": np.stack([l2a_mask.astype(np.uint8)] * 3, axis=-1)
        }
        # emit_image.georreference(arr, fill_value_default=True) → GeoTensor
        def _georef(arr, fill_value_default=True):
            return GeoTensor(arr, transform=transform, crs=crs,
                             fill_value_default=fill_value_default)
        emit_image.georreference.side_effect = _georef
    return emit_image


class TestComputeEmitMaskAugmentation:
    """compute_emit must build the right mask without calling the real LUT.

    We monkeypatch the inner ``compute`` to capture the mask it is called with
    instead of actually running plume vetting (which needs the LUT, fits, etc.).
    """

    def _patch_compute(self, monkeypatch):
        captured = {}

        def fake_compute(**kwargs):
            captured.update(kwargs)
            return {}

        monkeypatch.setattr(pv, "compute", fake_compute)
        return captured

    def _patch_target_signature(self, monkeypatch):
        # Skip the LUT path; compute_emit calls load_target_spectrum_mf internally.
        monkeypatch.setattr(
            pv, "load_target_spectrum_mf", lambda _img: np.ones(50)
        )

    def test_radiance_fill_pixels_are_masked(self, monkeypatch):
        captured = self._patch_compute(monkeypatch)
        self._patch_target_signature(monkeypatch)

        h, w, b = 20, 20, 50
        rdn = np.ones((h, w, b))
        rdn[3, 7, :] = pv.EMIT_RADIANCE_FILL_VALUE  # whole-pixel fill
        rdn[5, 5, 10] = pv.EMIT_RADIANCE_FILL_VALUE  # single-band fill triggers any-band rule
        transform = Affine(60.0, 0.0, 500000.0, 0.0, -60.0, 4000000.0)
        crs = "EPSG:32610"
        cmf = GeoTensor(np.zeros((h, w), dtype=np.float32),
                        transform=transform, crs=crs, fill_value_default=-9999.0)

        emit_image = _mock_emit_image(rdn, np.linspace(400, 2500, b), transform, crs)
        pv.compute_emit(emit_image, cmf, polygons=[], use_l2a_mask=False)

        mask = captured["clouds_and_surface_water_mask"]
        assert mask[3, 7] is np.True_ or mask[3, 7] == True  # noqa: E712
        assert mask[5, 5]
        # Pixels that are entirely clean stay unmasked
        assert not mask[0, 0]

    def test_radiance_nan_pixels_are_masked(self, monkeypatch):
        captured = self._patch_compute(monkeypatch)
        self._patch_target_signature(monkeypatch)

        h, w, b = 15, 15, 50
        rdn = np.ones((h, w, b))
        rdn[8, 8, 20] = np.nan
        transform = Affine(60.0, 0.0, 500000.0, 0.0, -60.0, 4000000.0)
        crs = "EPSG:32610"
        cmf = GeoTensor(np.zeros((h, w), dtype=np.float32),
                        transform=transform, crs=crs, fill_value_default=-9999.0)

        emit_image = _mock_emit_image(rdn, np.linspace(400, 2500, b), transform, crs)
        pv.compute_emit(emit_image, cmf, polygons=[], use_l2a_mask=False)

        assert captured["clouds_and_surface_water_mask"][8, 8]

    def test_cmf_fill_pixels_are_masked(self, monkeypatch):
        captured = self._patch_compute(monkeypatch)
        self._patch_target_signature(monkeypatch)

        h, w, b = 10, 10, 50
        rdn = np.ones((h, w, b))
        transform = Affine(60.0, 0.0, 500000.0, 0.0, -60.0, 4000000.0)
        crs = "EPSG:32610"
        cmf_vals = np.zeros((h, w), dtype=np.float32)
        cmf_vals[4, 4] = -9999.0  # explicit fill
        cmf_vals[6, 6] = np.nan
        cmf = GeoTensor(cmf_vals, transform=transform, crs=crs, fill_value_default=-9999.0)

        emit_image = _mock_emit_image(rdn, np.linspace(400, 2500, b), transform, crs)
        pv.compute_emit(emit_image, cmf, polygons=[], use_l2a_mask=False)

        mask = captured["clouds_and_surface_water_mask"]
        assert mask[4, 4]
        assert mask[6, 6]

    def test_use_l2a_mask_false_ignores_cloud_mask(self, monkeypatch):
        captured = self._patch_compute(monkeypatch)
        self._patch_target_signature(monkeypatch)

        h, w, b = 12, 12, 50
        rdn = np.ones((h, w, b))
        transform = Affine(60.0, 0.0, 500000.0, 0.0, -60.0, 4000000.0)
        crs = "EPSG:32610"
        cmf = GeoTensor(np.zeros((h, w), dtype=np.float32),
                        transform=transform, crs=crs, fill_value_default=-9999.0)

        # Cloud-flag every pixel in the L2A mask; the test asserts these are
        # ignored when use_l2a_mask=False.
        l2a_mask = np.ones((h, w), dtype=bool)
        emit_image = _mock_emit_image(rdn, np.linspace(400, 2500, b), transform, crs,
                                       l2a_mask=l2a_mask)

        pv.compute_emit(emit_image, cmf, polygons=[], use_l2a_mask=False)
        mask = captured["clouds_and_surface_water_mask"]
        # Mask should be all-False because no fill / NaN was injected and L2A is off.
        assert not mask.any()
        # nc_ds_l2amask should not have been accessed
        emit_image.georreference.assert_not_called()

    def test_use_l2a_mask_true_includes_cloud_mask(self, monkeypatch):
        captured = self._patch_compute(monkeypatch)
        self._patch_target_signature(monkeypatch)

        h, w, b = 12, 12, 50
        rdn = np.ones((h, w, b))
        transform = Affine(60.0, 0.0, 500000.0, 0.0, -60.0, 4000000.0)
        crs = "EPSG:32610"
        cmf = GeoTensor(np.zeros((h, w), dtype=np.float32),
                        transform=transform, crs=crs, fill_value_default=-9999.0)

        l2a_mask = np.zeros((h, w), dtype=bool)
        l2a_mask[2, 3] = True
        l2a_mask[7, 9] = True
        emit_image = _mock_emit_image(rdn, np.linspace(400, 2500, b), transform, crs,
                                       l2a_mask=l2a_mask)

        pv.compute_emit(emit_image, cmf, polygons=[], use_l2a_mask=True)
        mask = captured["clouds_and_surface_water_mask"]
        assert mask[2, 3]
        assert mask[7, 9]
        # Other pixels remain unmasked (radiance / cmf are clean)
        assert not mask[0, 0]

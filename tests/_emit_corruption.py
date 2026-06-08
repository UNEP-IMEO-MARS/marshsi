"""Shared test helper: inject non-finite values into EMIT radiance.

Used by the NaN-handling tests for ``mag1c_emit`` and ``AT_MF_total_EMIT`` (the
wmf processor). Both functions read radiance via
``read_from_bands(...).load_raw(transpose=False)``, so corrupting that single
call lets us exercise the real read/retrieval path with controlled bad pixels.
"""

import numpy as np


def inject_into_load_raw(monkeypatch, kinds):
    """Patch ``EMITImage.load_raw`` to corrupt a few currently-valid pixels of
    the ``(rows, cols, bands)`` array returned with ``transpose=False``.

    ``kinds`` is a list of corruption types, each applied to a distinct valid
    pixel: ``"nan"`` (whole pixel NaN), ``"nan_band"`` (NaN in one band) or
    ``"inf"`` (whole pixel +inf).

    Returns a list that, once ``load_raw`` has been called, holds the
    ``(row, col)`` positions that were corrupted.
    """
    from georeader.readers import emit

    orig = emit.EMITImage.load_raw
    recorded: list[tuple[int, int]] = []

    def patched(self, transpose=True):
        arr = np.asarray(orig(self, transpose=transpose)).astype(np.float64).copy()
        # Only corrupt the (rows, cols, bands) layout, and only once.
        if transpose or arr.ndim != 3 or recorded:
            return arr
        fill = self.fill_value_default
        valid = np.all(np.isfinite(arr) & (arr != fill), axis=-1)  # (rows, cols)
        ys, xs = np.where(valid)
        assert len(ys) >= len(kinds), "fixture has too few valid pixels"
        step = len(ys) // (len(kinds) + 1)
        nb = arr.shape[-1]
        for i, kind in enumerate(kinds):
            y, x = int(ys[(i + 1) * step]), int(xs[(i + 1) * step])
            if kind == "nan":
                arr[y, x, :] = np.nan
            elif kind == "nan_band":
                arr[y, x, nb // 2] = np.nan
            elif kind == "inf":
                arr[y, x, :] = np.inf
            else:  # pragma: no cover - guard against typos in tests
                raise ValueError(kind)
            recorded.append((y, x))
        return arr

    monkeypatch.setattr(emit.EMITImage, "load_raw", patched)
    return recorded

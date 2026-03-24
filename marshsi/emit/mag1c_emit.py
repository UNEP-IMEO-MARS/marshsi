# This code is an adapted version from Rutzika et al 2023:
# https://colab.research.google.com/drive/1hVcLoY3R0QryLVSLG6C4jTSG-4Jgg_4c?usp=sharing
# The original mag1c code is here: https://github.com/markusfoote/mag1c/blob/master/mag1c/mag1c.py#LL579C71-L579C83

from typing import Optional, Tuple, Union

import numpy as np

try:
    import torch
except ImportError:
    raise ImportError(
        "torch is required for mag1c. Install it with: pip install marshsi[mag1c]"
    )

from georeader.geotensor import GeoTensor
from georeader.readers.emit import EMITImage
from tqdm import tqdm

from . import mag1c

DEFAULT_WAVELENGTH_RANGE = (2122, 2488)


def mag1c_emit(
    ei: EMITImage,
    device: Optional[torch.device] = torch.device("cpu"),
    use_wavelength_range=DEFAULT_WAVELENGTH_RANGE,
    num_iter: int = 30,
    covariance_lerp_alpha: float = 1e-4,
    column_step: Optional[int] = None,
    georeferenced: bool = True,
    display_pbar: bool = True,
) -> Tuple[Union[GeoTensor, np.array], Union[GeoTensor, np.ndarray]]:
    """
    Run mag1c filter on an EMITImage object.

    Args:
        ei (EMITImage): EMITImage object
        device (Optional[torch.device], optional): device to use for the computation. Defaults to torch.device('cpu').
        use_wavelength_range (tuple, optional): wavelength range to use. Defaults to (2000, 2485).
        num_iter (int, optional): Number of iterations. Defaults to 30.
        covariance_lerp_alpha (float, optional): parameter to control the covariance matrix. Defaults to 1e-4.
        column_step (Optional[int], optional): column step to use. Defaults to None. Column step 1 runs the
            mag1c filter by column. None runs the mag1c filter in all the image.
        georeferenced (bool, optional): If True, the output is a GeoTensor. Defaults to True.
        display_pbar (bool, optional): If True, display a progress bar. Defaults to True.

    Returns:
        Tuple[Union[GeoTensor,np.array], Union[GeoTensor,np.array]]: Tuple with the mag1c filter and the albedo.

    """

    band_selection = (ei.wavelengths >= use_wavelength_range[0]) & (
        ei.wavelengths <= use_wavelength_range[1]
    )
    assert band_selection.any(), "There are no bands in the selected wavelength range"

    ei = ei.read_from_bands(band_selection)

    target = mag1c.generate_template_from_bands(centers=ei.wavelengths, fwhm=ei.fwhm)
    target = target.astype(np.float64)

    spec = torch.tensor(target[:, 1], device=device)

    raw_data = ei.load_raw(transpose=False)  # (rows, cols, bands)
    invalid = np.any(raw_data == ei.fill_value_default, axis=-1)  # (rows, cols)

    mag1c_output = np.full(
        invalid.shape, dtype=np.float64, fill_value=ei.fill_value_default
    )  # (rows, cols)
    albedo_out = np.full(
        invalid.shape, dtype=np.float64, fill_value=ei.fill_value_default
    )  # (rows, cols)

    column_step = column_step or raw_data.shape[1]

    listrange = range(0, raw_data.shape[1], column_step)
    for columnstart in tqdm(
        listrange,
        total=len(listrange),
        desc="\tRunning mag1c filter by columns",
        disable=not display_pbar or (column_step == raw_data.shape[1]),
    ):
        columnend = min(columnstart + column_step, raw_data.shape[1])
        invalid_slice = invalid[:, columnstart:columnend]  # (rows, column_step)
        raw_data_slice = raw_data[:, columnstart:columnend, :]  # (rows, column_step, bands)

        valid_slice = ~invalid_slice
        if not valid_slice.any():
            continue

        raw_data_slice = raw_data_slice[valid_slice, :][np.newaxis]  # (batch, pixels, bands)
        # pixels is the number of valid pixels (np.sum(valid_slice))

        # Convert to float64 to avoid rounding errors
        raw_data_slice = raw_data_slice.astype(np.float64)
        raw_data_slice = torch.tensor(raw_data_slice, device=device)

        mf_out, alb_out = mag1c.acrwl1mf(
            raw_data_slice, spec, num_iter=num_iter, alpha=covariance_lerp_alpha
        )

        mf_out = np.array(mf_out.cpu())[0, :, 0]  # (pixels,)
        alb_out = np.array(alb_out.cpu())[0, :, 0]  # (pixels,)

        mag1c_output[:, columnstart:columnend][valid_slice] = mf_out
        albedo_out[:, columnstart:columnend][valid_slice] = alb_out

    if georeferenced:
        mag1c_output = ei.georreference(mag1c_output, fill_value_default=ei.fill_value_default)
        albedo_out = ei.georreference(albedo_out, fill_value_default=ei.fill_value_default)

    return mag1c_output.astype(np.float32), albedo_out.astype(np.float32)

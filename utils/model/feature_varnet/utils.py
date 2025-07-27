import math
import torch
from torch import Tensor
from typing import List, NamedTuple, Optional, Tuple
from fastmri.data.transforms import center_crop
from fastmri.fftc import fft2c_new as fft2c, ifft2c_new as ifft2c
from fastmri.math import complex_abs, complex_conj, complex_mul


def image_crop(image: Tensor, crop_size: Optional[Tuple[int, int]] = None) -> Tensor:
    if crop_size is None:
        return image
    return center_crop(image, crop_size).contiguous()


def _calc_uncrop(crop_height: int, in_height: int) -> Tuple[int, int]:
    pad_height = (in_height - crop_height) // 2
    if (in_height - crop_height) % 2 != 0:
        pad_height_top = pad_height + 1
    else:
        pad_height_top = pad_height

    pad_height = in_height - pad_height

    return pad_height_top, pad_height


def image_uncrop(image: Tensor, original_image: Tensor) -> Tensor:
    """Insert values back into original image."""
    in_shape = original_image.shape
    original_image = original_image.clone()

    if in_shape == image.shape:
        return image

    pad_height_top, pad_height = _calc_uncrop(image.shape[-2], in_shape[-2])
    pad_height_left, pad_width = _calc_uncrop(image.shape[-1], in_shape[-1])

    try:
        if len(in_shape) == 2:  # Assuming 2D images
            original_image[pad_height_top:pad_height, pad_height_left:pad_width] = image
        elif len(in_shape) == 3:  # Assuming 3D images with channels
            original_image[
                :, pad_height_top:pad_height, pad_height_left:pad_width
            ] = image
        elif len(in_shape) == 4:  # Assuming 4D images with batch size
            original_image[
                :, :, pad_height_top:pad_height, pad_height_left:pad_width
            ] = image
        else:
            raise RuntimeError(f"Unsupported tensor shape: {in_shape}")
    except RuntimeError:
        print(f"in_shape: {in_shape}, image shape: {image.shape}")
        raise

    return original_image


def norm_fn(image: Tensor, means: Tensor, variances: Tensor) -> Tensor:
    means = means.view(1, -1, 1, 1)
    variances = variances.view(1, -1, 1, 1)
    return (image - means) * torch.rsqrt(variances)


def unnorm_fn(image: Tensor, means: Tensor, variances: Tensor) -> Tensor:
    means = means.view(1, -1, 1, 1)
    variances = variances.view(1, -1, 1, 1)
    return image * torch.sqrt(variances) + means


def complex_to_chan_dim(x: Tensor) -> Tensor:
    b, c, h, w, two = x.shape
    assert two == 2
    assert c == 1
    return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)


def chan_complex_to_last_dim(x: Tensor) -> Tensor:
    b, c2, h, w = x.shape
    assert c2 == 2
    c = c2 // 2
    return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()


def sens_expand(x: Tensor, sens_maps: Tensor) -> Tensor:
    return fft2c(complex_mul(chan_complex_to_last_dim(x), sens_maps))


def sens_reduce(x: Tensor, sens_maps: Tensor) -> Tensor:
    return complex_to_chan_dim(
        complex_mul(ifft2c(x), complex_conj(sens_maps)).sum(dim=1, keepdim=True)
    )

class FeatureImage(NamedTuple):
    features: Tensor
    sens_maps: Optional[Tensor] = None
    crop_size: Optional[Tuple[int, int]] = None
    means: Optional[Tensor] = None
    variances: Optional[Tensor] = None
    mask: Optional[Tensor] = None
    ref_kspace: Optional[Tensor] = None
    beta: Optional[Tensor] = None   # ← PSF offset look-up 테이블
    gamma: Optional[Tensor] = None

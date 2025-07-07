import numpy as np
import torch

import numpy.fft as fft
import h5py
import os
from tqdm import tqdm
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import fastmri
from .subsample import MaskFunc


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, kspace, target, maximum, fname, slice







def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()


def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    offset: Optional[int] = None,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_frequencies = mask_func(shape, offset, seed)
    if padding is not None:
        mask[..., : padding[0], :] = 0
        mask[..., padding[1] :, :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, num_low_frequencies


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask


def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Can operate with different masks for each batch element.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError("mask_from and mask_to must match shapes.")
    if not mask_from.ndim == 1:
        raise ValueError("mask_from and mask_to must have 1 dimension.")
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
            not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError("mask_from and mask_to must have batch_size length.")

    if mask_from.shape[0] == 1:
        mask = mask_center(x, int(mask_from), int(mask_to))
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, :, :, start:end] = x[i, :, :, start:end]

    return mask


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]


def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


class UnetSample(NamedTuple):
    """
    A subsampled image for U-Net reconstruction.

    Args:
        image: Subsampled image after inverse FFT.
        target: The target image (if applicable).
        mean: Per-channel mean values used for normalization.
        std: Per-channel standard deviations used for normalization.
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
    """

    image: torch.Tensor
    target: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    fname: str
    slice_num: int
    max_value: float


class UnetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """
        kspace_torch = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            # we only need first element, which is k-space after masking
            masked_kspace = apply_mask(kspace_torch, self.mask_func, seed=seed)[0]
        else:
            masked_kspace = kspace_torch

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        if target is not None:
            crop_size = (target.shape[-2], target.shape[-1])
        else:
            crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        if image.shape[-2] < crop_size[1]:
            crop_size = (image.shape[-2], image.shape[-2])

        image = complex_center_crop(image, crop_size)

        # absolute value
        image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "multicoil":
            image = fastmri.rss(image)

        # normalize input
        image, mean, std = normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target_torch = to_tensor(target)
            target_torch = center_crop(target_torch, crop_size)
            target_torch = normalize(target_torch, mean, std, eps=1e-11)
            target_torch = target_torch.clamp(-6, 6)
        else:
            target_torch = torch.Tensor([0])

        return UnetSample(
            image=image,
            target=target_torch,
            mean=mean,
            std=std,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
        )


class VarNetSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """

    masked_kspace: torch.Tensor
    mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]


class VarNetDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: Optional[np.ndarray],
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> VarNetSample:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A VarNetSample with the masked k-space, sampling mask, target
            image, the filename, the slice number, the maximum image value
            (from target), the target crop size, and the number of low
            frequency lines sampled.
        """
        if target is not None:
            target_torch = to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

            sample = VarNetSample(
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=num_low_frequencies,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
            )
        else:
            masked_kspace = kspace_torch
            shape = np.array(kspace_torch.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask_torch = mask_torch.reshape(*mask_shape)
            mask_torch[:, :, :acq_start] = 0
            mask_torch[:, :, acq_end:] = 0

            sample = VarNetSample(
                masked_kspace=masked_kspace,
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=0,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
            )

        return sample


class MiniCoilSample(NamedTuple):
    """
    A sample of masked coil-compressed k-space for reconstruction.

    Args:
        kspace: the original k-space before masking.
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """

    kspace: torch.Tensor
    masked_kspace: torch.Tensor
    mask: torch.Tensor
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]


class MiniCoilTransform:
    """
    Multi-coil compressed transform, for faster prototyping.
    """

    def __init__(
        self,
        mask_func: Optional[MaskFunc] = None,
        use_seed: Optional[bool] = True,
        crop_size: Optional[tuple] = None,
        num_compressed_coils: Optional[int] = None,
    ):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
            crop_size: Image dimensions for mini MR images.
            num_compressed_coils: Number of coils to output from coil
                compression.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.crop_size = crop_size
        self.num_compressed_coils = num_compressed_coils

    def __call__(self, kspace, mask, target, attrs, fname, slice_num):
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset. Not used if mask_func is defined.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                kspace: original kspace (used for active acquisition only).
                masked_kspace: k-space after applying sampling mask. If there
                    is no mask or mask_func, returns same as kspace.
                mask: The applied sampling mask
                target: The target image (if applicable). The target is built
                    from the RSS opp of all coils pre-compression.
                fname: File name.
                slice_num: The slice index.
                max_value: Maximum image value.
                crop_size: The size to crop the final image.
        """
        if target is not None:
            target = to_tensor(target)
            max_value = attrs["max"]
        else:
            target = torch.tensor(0)
            max_value = 0.0

        if self.crop_size is None:
            crop_size = torch.tensor([attrs["recon_size"][0], attrs["recon_size"][1]])
        else:
            if isinstance(self.crop_size, tuple) or isinstance(self.crop_size, list):
                assert len(self.crop_size) == 2
                if self.crop_size[0] is None or self.crop_size[1] is None:
                    crop_size = torch.tensor(
                        [attrs["recon_size"][0], attrs["recon_size"][1]]
                    )
                else:
                    crop_size = torch.tensor(self.crop_size)
            elif isinstance(self.crop_size, int):
                crop_size = torch.tensor((self.crop_size, self.crop_size))
            else:
                raise ValueError(
                    f"`crop_size` should be None, tuple, list, or int, not: {type(self.crop_size)}"
                )

        if self.num_compressed_coils is None:
            num_compressed_coils = kspace.shape[0]
        else:
            num_compressed_coils = self.num_compressed_coils

        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = 0
        acq_end = crop_size[1]

        # new cropping section
        square_crop = (attrs["recon_size"][0], attrs["recon_size"][1])
        kspace = fastmri.fft2c(
            complex_center_crop(fastmri.ifft2c(to_tensor(kspace)), square_crop)
        ).numpy()
        kspace = complex_center_crop(kspace, crop_size)

        # we calculate the target before coil compression. This causes the mini
        # simulation to be one where we have a 15-coil, low-resolution image
        # and our reconstructor has an SVD coil approximation. This is a little
        # bit more realistic than doing the target after SVD compression
        target = fastmri.rss_complex(fastmri.ifft2c(to_tensor(kspace)))
        max_value = target.max()

        # apply coil compression
        new_shape = (num_compressed_coils,) + kspace.shape[1:]
        kspace = np.reshape(kspace, (kspace.shape[0], -1))
        left_vec, _, _ = np.linalg.svd(kspace, compute_uv=True, full_matrices=False)
        kspace = np.reshape(
            np.array(np.matrix(left_vec[:, :num_compressed_coils]).H @ kspace),
            new_shape,
        )
        kspace = to_tensor(kspace)

        # Mask kspace
        if self.mask_func:
            masked_kspace, mask, _ = apply_mask(
                kspace, self.mask_func, seed, (acq_start, acq_end)
            )
            mask = mask.byte()
        elif mask is not None:
            masked_kspace = kspace
            shape = np.array(kspace.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask = mask.reshape(*mask_shape)
            mask = mask.byte()
        else:
            masked_kspace = kspace
            shape = np.array(kspace.shape)
            num_cols = shape[-2]

        return MiniCoilSample(
            kspace, masked_kspace, mask, target, fname, slice_num, max_value, crop_size
        )








import torch
import numpy as np
import numpy.fft as fft
from typing import Dict, NamedTuple, Optional, Tuple

# 결과를 담을 NamedTuple을 정의합니다.
class GccSample(NamedTuple):
    kspace: torch.Tensor
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]

class GCCTransform:
    """
    GCC 압축 및 크롭을 실시간(on-the-fly)으로 적용하는 Transform 클래스.
    PyTorch DataLoader 파이프라인에 직접 사용할 수 있습니다.
    """
    def __init__(self,
                 target_coils: int = 4,
                 num_calib_lines: int = 24,
                 sliding_window_size: int = 5,
                 use_aligned_gcc: bool = True,
                 crop_size: Optional[Tuple[int, int]] = (320, 320)):
        """
        GCCTransform 클래스의 생성자.

        Args:
            target_coils (int): 압축 후 목표 가상 코일의 수.
            num_calib_lines (int): k-space 중앙에서 추출할 보정 데이터 라인(PE)의 수.
            sliding_window_size (int): GCC 행렬 계산 시 사용할 슬라이딩 윈도우의 크기.
            use_aligned_gcc (bool): 정렬된(aligned) GCC 압축 행렬을 사용할지 여부.
            crop_size (Optional[Tuple[int, int]]): 압축 후 크롭할 이미지 크기.
                                                     None이면 크롭하지 않음.
        """
        self.target_coils = target_coils
        self.num_calib_lines = num_calib_lines
        self.sliding_window_size = sliding_window_size
        self.use_aligned_gcc = use_aligned_gcc
        self.crop_size = crop_size
        self.compression_dim = 1  # GCC는 Readout(RO) 방향으로 적용되므로 고정

    def _calc_gcc_matrices(self, calib_data: np.ndarray) -> np.ndarray:
        """(비공개) 보정 데이터로부터 GCC 압축 행렬을 계산합니다."""
        # (PE, RO, Coils) -> (RO, PE, Coils)
        calib_data_transposed = calib_data.transpose(1, 0, 2)
        
        # Readout 축(이제 첫 번째 축)을 따라 iFFT를 수행하여 하이브리드 공간으로 변환
        hybrid_calib_data = fft.ifft(fft.ifftshift(calib_data_transposed, axes=0), axis=0)
        
        n_readout, _, n_coils = hybrid_calib_data.shape
        gcc_matrices = np.zeros((n_readout, n_coils, n_coils), dtype=np.complex128)
        
        for i in range(n_readout):
            start = max(0, i - self.sliding_window_size // 2)
            end = min(n_readout, i + self.sliding_window_size // 2 + 1)
            
            block = hybrid_calib_data[start:end, :, :]
            reshaped_block = block.reshape(-1, n_coils)
            _, _, vh = np.linalg.svd(reshaped_block, full_matrices=False)
            gcc_matrices[i, :, :] = vh.conj().T
            
        return gcc_matrices

    def _align_gcc_matrices(self, gcc_matrices: np.ndarray) -> np.ndarray:
        """(비공개) 계산된 GCC 행렬들을 정렬합니다."""
        n_readout, n_coils, _ = gcc_matrices.shape
        cropped_matrices = gcc_matrices[:, :, :self.target_coils]
        aligned_matrices = np.zeros_like(cropped_matrices, dtype=np.complex128)
        aligned_matrices[0] = cropped_matrices[0]
        
        for i in range(1, n_readout):
            prev_aligned_v = aligned_matrices[i - 1]
            current_v = cropped_matrices[i]
            correlation_matrix = prev_aligned_v.T.conj() @ current_v
            u_rot, _, vh_rot = np.linalg.svd(correlation_matrix, full_matrices=False)
            rotation_matrix = u_rot @ vh_rot
            aligned_matrices[i] = current_v @ rotation_matrix
            
        return aligned_matrices

    def _apply_compression(self, kspace_data: np.ndarray, compression_matrices: np.ndarray) -> np.ndarray:
        """(비공개) 압축 행렬을 적용합니다."""
        # kspace(p,r,c)와 matrix(r,c,v)를 곱해 compressed_kspace(p,r,v) 생성
        return np.einsum('prc, rcv -> prv', kspace_data, compression_matrices)

    def __call__(self, kspace: np.ndarray, target: np.ndarray, attrs: Dict, fname: str, slice_num: int) -> GccSample:
        """
        단일 k-space 슬라이스에 GCC 압축 및 크롭을 적용합니다.

        Args:
            kspace (np.ndarray): (Coils, PE, RO) 형태의 복소수 k-space 슬라이스.
            target (np.ndarray): 정답 이미지.
            attrs (Dict): h5 파일의 속성 정보.
            fname (str): 파일 이름.
            slice_num (int): 슬라이스 번호.

        Returns:
            GccSample: 압축 및 변환이 완료된 데이터 샘플.
        """
        # 1. k-space를 (PE, RO, Coils) 형태로 변환
        kspace_slice_3d = kspace.transpose(1, 2, 0)

        # 2. 보정 데이터 추출
        n_pe = kspace_slice_3d.shape[0]
        calib_start = n_pe // 2 - self.num_calib_lines // 2
        calib_end = n_pe // 2 + self.num_calib_lines // 2
        calib_data = kspace_slice_3d[calib_start:calib_end, :, :]

        # 3. GCC 행렬 계산
        gcc_matrices_full = self._calc_gcc_matrices(calib_data)

        # 4. 전체 k-space를 하이브리드 공간(x, k_y)으로 변환
        kspace_transposed = kspace_slice_3d.transpose(1, 0, 2)
        hybrid_space_full = fft.ifft(fft.ifftshift(kspace_transposed, axes=0), axis=0)
        hybrid_space_for_einsum = hybrid_space_full.transpose(1, 0, 2)

        # 5. 압축 행렬 선택 및 적용
        if self.use_aligned_gcc:
            compression_matrices = self._align_gcc_matrices(gcc_matrices_full)
        else:
            compression_matrices = gcc_matrices_full[:, :, :self.target_coils]
        
        compressed_hybrid = self._apply_compression(hybrid_space_for_einsum, compression_matrices)

        # 6. 다시 k-space로 변환
        kspace_compressed_shifted = fft.fft(compressed_hybrid, axis=1)
        compressed_kspace_np = fft.fftshift(kspace_compressed_shifted, axes=1) # (PE, RO, VCoils)

        # 7. PyTorch 텐서로 변환 (Coils, PE, RO)
        kspace_torch = to_tensor(compressed_kspace_np.transpose(2, 0, 1))

        # 8. (선택적) 최종 크기로 크롭
        if self.crop_size is not None:
            # 타겟 이미지 생성 (압축 전, 크롭 전 k-space 사용)
            # RSS 이미지를 생성하여 크롭하고, max_value를 attrs에 업데이트
            pre_crop_image = fastmri.rss(fastmri.ifft2c(to_tensor(kspace)))
            target_image = center_crop(pre_crop_image.unsqueeze(0), self.crop_size).squeeze(0)
            max_value = target_image.max()
            
            # 압축된 k-space를 이미지로 변환 후 크롭
            image_compressed = fastmri.ifft2c(kspace_torch)
            image_cropped = complex_center_crop(image_compressed, self.crop_size)
            kspace_final = fastmri.fft2c(image_cropped)
        else:
            # 크롭을 안 할 경우, crop_size를 원본 크기로 설정
            self.crop_size = (attrs['recon_size'][0], attrs['recon_size'][1])
            target_image = to_tensor(target)
            max_value = attrs['max']
            kspace_final = kspace_torch
        
        return GccSample(
            kspace=kspace_final,
            target=target_image,
            fname=fname,
            slice_num=slice_num,
            max_value=max_value,
            crop_size=self.crop_size,
        )

# --- 사용 예시 ---
if __name__ == '__main__':
    # 가상의 Transform 객체 생성
    gcc_transform = GCCTransform(
        target_coils=4,
        num_calib_lines=24,
        use_aligned_gcc=True,
        crop_size=(320, 320)
    )

    # 가상의 입력 데이터 생성 (실제 데이터셋에서 제공되는 형태)
    dummy_kspace = (np.random.randn(15, 372, 320) + 1j * np.random.randn(15, 372, 320)).astype(np.complex64)
    dummy_target = np.random.randn(320, 320).astype(np.float32)
    dummy_attrs = {'max': 1.0, 'recon_size': (372, 320)}
    dummy_fname = "dummy_file"
    dummy_slice_num = 0

    # Transform 적용
    transformed_sample = gcc_transform(
        kspace=dummy_kspace,
        target=dummy_target,
        attrs=dummy_attrs,
        fname=dummy_fname,
        slice_num=dummy_slice_num
    )

    # 결과 확인
    print("GCC Transform 적용 완료!")
    print(f"압축/크롭 후 k-space 크기: {transformed_sample.kspace.shape}")
    print(f"크롭 후 target 크기: {transformed_sample.target.shape}")
    print(f"가상 코일 수: {transformed_sample.kspace.shape[0]}")
    print(f"최종 H, W 크기: {transformed_sample.kspace.shape[-2:]}")
    print(f"Sample 정보: {transformed_sample}")
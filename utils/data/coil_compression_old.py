import numpy as np
import torch
import numpy.fft as fft
from tqdm import tqdm
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union
from utils.data.transforms import to_tensor
from utils.data.subsample import MaskFunc
from utils.data.transforms_Facebook import center_crop, complex_center_crop, apply_mask
import fastmri

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
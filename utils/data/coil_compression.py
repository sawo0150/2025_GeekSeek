# utils/data/coil_compression.py

from abc import ABC, abstractmethod
import numpy as np
import torch

class BaseCompressor(ABC):
    """모든 compressor는 __call__(kspace, attrs) → (kspace_compressed, attrs_upd) 형태로 동작."""
    def __init__(self, target_coils: int):
        self.target_coils = target_coils

    @abstractmethod
    def compress(self, kspace: np.ndarray, attrs: dict) -> np.ndarray:
        ...

    def __call__(self, mask, kspace, target, attrs, fname, slice_num):
        # 1) numpy로 받고
        kspace_np = kspace.numpy() if isinstance(kspace, torch.Tensor) else kspace
        # 2) 실제 압축
        if kspace_np.ndim == 4 and kspace_np.shape[-1] == 2:
            # real = kspace[..., 0], imag = kspace[..., 1]
            kspace_np = kspace_np[...,0] + 1j * kspace_np[...,1]
        kspace_cmp = self.compress(kspace_np, attrs)
        # 3) torch Tensor로 복귀
        kspace_t_complex = torch.from_numpy(kspace_cmp)
        # 수정) 모델이 기대하는 real/imag 포맷으로 변환 
        kspace_t_real_imag = torch.view_as_real(kspace_t_complex)
        # 4) 이후 기존 DataTransform이 기대하는 튜플 형태로 반환
        return mask, kspace_t_real_imag, target, attrs, fname, slice_num

class IdentityCompressor(BaseCompressor):
    """압축을 전혀 하지 않고, 입력 k-space를 그대로 반환합니다."""
    def __init__(self):
        # target_coils는 필요 없으니 그냥 0이나 None
        super().__init__(target_coils=0)

    def compress(self, kspace: np.ndarray, attrs: dict) -> np.ndarray:
        return kspace
    
class SCCCompressor(BaseCompressor):
    def __init__(self, target_coils: int = 4, num_calib_lines: int = 24):
        super().__init__(target_coils)
        self.num_calib_lines = num_calib_lines

    # @profile  # <-- line_profiler 이 읽는 어노테이션
    def compress(self, kspace: np.ndarray, attrs: dict) -> np.ndarray:
        # ┕ [수정] kspace 전체 대신 중앙의 보정 라인(calibration lines)을 사용하여 SVD를 수행
        # kspace shape: (C, H, W)
        C, H, W = kspace.shape

        # 1) 보정 데이터 추출 (k-space의 중앙)
        calib_start = H // 2 - self.num_calib_lines // 2
        calib_end = H // 2 + self.num_calib_lines // 2
        calib_data = kspace[:, calib_start:calib_end, :] # (C, num_calib_lines, W)

        # 2) 보정 데이터를 펼쳐서 SVD 수행
        calib_flat = calib_data.reshape(C, -1) # (C, num_calib_lines * W)
        u, s, vh = np.linalg.svd(calib_flat, full_matrices=False)

        # 3) 상위 self.target_coils 개수에 해당하는 압축 행렬 생성
        u_reduced = u[:, : self.target_coils] # (C, T)

        # 4) 생성된 압축 행렬을 '전체' k-space에 적용
        kspace_flat = kspace.reshape(C, -1)             # (C, H*W)
        # 켤레 전치(Hermitian transpose)를 사용하여 올바르게 압축
        compressed_flat = u_reduced.conj().T @ kspace_flat     # (T, H*W) # ✅ 수정된 라인

        # 5) 원래 형태로 복원 (T, H, W)
        compressed = compressed_flat.reshape(self.target_coils, H, W)

        # 6) 타입 복원 (optional)
        return compressed.astype(kspace.dtype)




class GCCCompressor(BaseCompressor):
    """GCC 압축을 수행하는 Compressor."""
    def __init__(self,
                 target_coils: int = 4,
                 num_calib_lines: int = 24,
                 sliding_window_size: int = 5,
                 use_aligned_gcc: bool = True):
        super().__init__(target_coils)
        self.num_calib_lines = num_calib_lines
        self.sliding_window_size = sliding_window_size
        self.use_aligned_gcc = use_aligned_gcc

    # GCCTransform에 있던 비공개 메서드들을 그대로 가져옵니다.
    def _calc_gcc_matrices(self, calib_data: np.ndarray) -> np.ndarray:
        # ... (GCCTransform의 _calc_gcc_matrices 코드 내용 전체 복사) ...
        # (PE, RO, Coils) -> (RO, PE, Coils)
        calib_data_transposed = calib_data.transpose(1, 0, 2)
        # Readout 축(이제 첫 번째 축)을 따라 iFFT를 수행하여 하이브리드 공간으로 변환
        hybrid_calib_data = np.fft.ifft(np.fft.ifftshift(calib_data_transposed, axes=0), axis=0)
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
        # ... (GCCTransform의 _align_gcc_matrices 코드 내용 전체 복사) ...
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
        # ... (GCCTransform의 _apply_compression 코드 내용 전체 복사) ...
        return np.einsum('prc, rcv -> prv', kspace_data, compression_matrices)

    def compress(self, kspace: np.ndarray, attrs: dict) -> np.ndarray:
        """
        BaseCompressor의 추상 메서드를 구현합니다.
        kspace: (Coils, PE, RO) 형태의 numpy 배열
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
        hybrid_space_full = np.fft.ifft(np.fft.ifftshift(kspace_transposed, axes=0), axis=0)
        hybrid_space_for_einsum = hybrid_space_full.transpose(1, 0, 2)
        # 5. 압축 행렬 선택 및 적용
        if self.use_aligned_gcc:
            compression_matrices = self._align_gcc_matrices(gcc_matrices_full)
        else:
            compression_matrices = gcc_matrices_full[:, :, :self.target_coils]
        compressed_hybrid = self._apply_compression(hybrid_space_for_einsum, compression_matrices)
        # 6. 다시 k-space로 변환
        kspace_compressed_shifted = np.fft.fft(compressed_hybrid, axis=1)
        compressed_kspace_np = np.fft.fftshift(kspace_compressed_shifted, axes=1) # (PE, RO, VCoils)
        # 7. (Coils, PE, RO) 형태로 최종 반환
        return compressed_kspace_np.transpose(2, 0, 1).astype(kspace.dtype)

# AlignedGCCCompressor는 이제 GCCCompressor의 옵션이 되었으므로 삭제하거나,
# 아래와 같이 간단한 래퍼로 남겨둘 수 있습니다.
class AlignedGCCCompressor(GCCCompressor):
    def __init__(self, target_coils: int = 4, num_calib_lines: int = 24, sliding_window_size: int = 5):
        super().__init__(target_coils, num_calib_lines, sliding_window_size, use_aligned_gcc=True)


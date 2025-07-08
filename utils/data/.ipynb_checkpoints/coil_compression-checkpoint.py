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
        kspace_t = torch.from_numpy(kspace_cmp)
        # 4) 이후 기존 DataTransform이 기대하는 튜플 형태로 반환
        return mask, kspace_t, target, attrs, fname, slice_num

class IdentityCompressor(BaseCompressor):
    """압축을 전혀 하지 않고, 입력 k-space를 그대로 반환합니다."""
    def __init__(self):
        # target_coils는 필요 없으니 그냥 0이나 None
        super().__init__(target_coils=0)

    def compress(self, kspace: np.ndarray, attrs: dict) -> np.ndarray:
        return kspace
    
class SCCCompressor(BaseCompressor):
    def __init__(self, target_coils: int = 4):
        super().__init__(target_coils)

    # @profile  # <-- line_profiler 이 읽는 어노테이션
    def compress(self, kspace: np.ndarray, attrs: dict) -> np.ndarray:
        # 1) (C, H, W) → (C, H*W)
        C, H, W = kspace.shape
        flat = kspace.reshape(C, -1)
        # 2) SVD
        u, s, vh = np.linalg.svd(flat, full_matrices=False)
        # 3) 상위 self.target_coils 개 프로젝션
        u_reduced = u[:, : self.target_coils]            # (C, T)
        compressed_flat = u_reduced.T @ flat             # (T, H*W)
        # 4) (T, H, W) 로 복원
        compressed = compressed_flat.reshape(self.target_coils, H, W)
        # 5) 타입 복원 (optional)
        return compressed.astype(kspace.dtype)



class GCCCompressor(BaseCompressor):
    def __init__(self, target_coils: int = 4, num_calib_lines: int = 24, sliding_window_size: int = 5):
        super().__init__(target_coils)
        self.num_calib_lines = num_calib_lines
        self.sliding_window_size = sliding_window_size
    def compress(self, kspace, attrs):
        # 여기엔 본인이 작성하신 GCC 알고리즘 코드
        # _calc_gcc_matrices, _align_gcc_matrices 등을 적용한 뒤
        # return compressed_kspace (np.ndarray)
        ...


class AlignedGCCCompressor(GCCCompressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def compress(self, kspace, attrs):
        # GCCCompressor.compress 에서 aligned 옵션만 켜놓은 버전
        ...



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
    def compress(self, kspace, attrs):
        # 간단 SVD-based coil compression 구현
        c, h, w = kspace.shape
        flat = kspace.reshape(c, -1)
        u, s, vh = np.linalg.svd(flat, full_matrices=False)
        compressed = (u[:, : self.target_coils].T @ flat).reshape(
            self.target_coils, h, w
        )
        return compressed


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



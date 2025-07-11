import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
from collections.abc import Mapping     #  ← 추가
import fastmri
 


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data)

# 공통 헬퍼: complex64 → real/imag 마지막 차원 텐서
def to_realimag(x: torch.Tensor) -> torch.Tensor:
    # x.dtype.is_complex 은 PyTorch 1.9 이상
    if x.dtype.is_complex:
        return torch.view_as_real(x)         # (..., H, W) → (..., H, W, 2)
    # 이미 real/imag 채널 포맷이라면 그대로
    if x.ndim >= 4 and x.shape[-1] == 2:
        return x
    raise ValueError(f"to_realimag: unexpected tensor shape/dtype {x.shape}, {x.dtype}")

# ---- CPU 上 토치 텐서 전용 FFT/티칭 함수 ----
def torch_ifft2c(x: torch.Tensor):
    # # x: (C, H, W, 2) real/imag
    # real, imag = x.unbind(-1)
    # cplx = torch.complex(real, imag)
    # out = fastmri.ifft2c(cplx)  # 혹은 직접 torch.fft.ifft2
    # return torch.stack((out.real, out.imag), dim=-1)
    # x: (C, H, W, 2) real/imag 채널로 들어왔다고 가정
    # 바로 fastmri.ifft2c에 넘기면 내부에서 view_as_complex 처리

    xr = to_realimag(x)
    out = fastmri.ifft2c(xr)  
    return out  # 이미 (..., H, W, 2) 포맷

def torch_fft2c(x: torch.Tensor):
    # real, imag = x.unbind(-1)
    # cplx = torch.complex(real, imag)
    # out = fastmri.fft2c(cplx)
    # return torch.stack((out.real, out.imag), dim=-1)

    # x: (C, H, W, 2) real/imag
    xr = to_realimag(x)
    out = fastmri.fft2c(xr)
    return out      # 내부 view_as_complex → FFT → view_as_real 리턴

# 파일: utils/data/transforms.py (or new 파일)
class MaskApplyTransform:
    """mask * kspace 만 수행하고, numpy 복소 k-space 리턴."""
    def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
        # ❶ numpy → complex64 tensor
        k_cplx = to_tensor(kspace).to(torch.complex64)  # (C, H, W), complex dtype
        m = torch.from_numpy(mask).float()   # shape (W)
        # ❷ undersample 적용 (브로드캐스트)
        k = k_cplx * m
        return m, k, target, attrs, fname, slice_idx

class CenterCropOrPad:
    """
    (C, H, W) 복소 k-space를 이미지 도메인에서 crop하거나,
    배경 노이즈를 추정하여 noise padding을 수행합니다.
    항상 (C, H0, W0) 크기를 반환합니다.
    """
    def __init__(self, target_size: Tuple[int,int], corner_size: int = 16):
        self.H0, self.W0 = target_size
        # ┕ [추가] 노이즈 추정에 사용할 코너 영역의 크기
        self.corner_size = corner_size

    # ┕ [추가] 이미지 코너에서 노이즈 표준편차를 추정하는 헬퍼 메서드
    def _estimate_noise_std(self, image_slice: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        이미지 네 귀퉁이에서 real/imag 채널의 노이즈 표준편차를 각각 계산합니다.
        image_slice: (C, H, W, 2) 형태의 텐서
        """
        C, H, W, _ = image_slice.shape
        # 코너 크기가 이미지 크기보다 크지 않도록 보정
        cs = min(self.corner_size, H // 2, W // 2)
        if cs == 0: # 이미지가 너무 작으면 표준편차 0으로 간주
            return torch.tensor(0.0), torch.tensor(0.0)

        # 네 귀퉁이 영역을 잘라 리스트에 추가
        corners = [
            image_slice[:, :cs, :cs, :],      # Top-left
            image_slice[:, :cs, -cs:, :],     # Top-right
            image_slice[:, -cs:, :cs, :],     # Bottom-left
            image_slice[:, -cs:, -cs:, :],    # Bottom-right
        ]
        
        # 모든 코너를 하나의 텐서로 결합 (W 차원을 따라)
        all_corners = torch.cat(corners, dim=2)
        
        # real/imag 채널 각각의 표준편차 계산
        std_real = torch.std(all_corners[..., 0])
        std_imag = torch.std(all_corners[..., 1])
        
        return std_real, std_imag

    def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
        # ── 0. attrs 가 dict/Mapping 아니면 빈 dict 로 대체 ─────────
        if not isinstance(attrs, Mapping):
            attrs = {}
        # kspace: torch.Tensor (C,H,W) complex represented as complex dtypece.real,kspace.imag),-1)
        # ❶ 입력을 complex64 dtype 으로 맞춤
        k_cplx = to_tensor(kspace).to(torch.complex64)
        k_ri = torch.view_as_real(k_cplx)
        img = torch_ifft2c(k_ri)

        C, H, W, _ = img.shape
        Hc, Wc = min(H, self.H0), min(W, self.W0)

        # 1. 크롭 (항상 수행)
        top, left = (H - Hc) // 2, (W - Wc) // 2
        img_cropped = img[:, top:top+Hc, left:left+Wc, :]

        # 2. 패딩 (필요시에만 노이즈로 수행)
        pad_h, pad_w = self.H0 - Hc, self.W0 - Wc
        
        # ┕ [수정] 기존 F.pad 로직을 노이즈 패딩으로 대체
        if pad_h > 0 or pad_w > 0:
            # 2a. 이미지 배경에서 노이즈 레벨 추정
            std_real, std_imag = self._estimate_noise_std(img)

            # 2b. 목표 크기의 텐서를 가우시안 노이즈로 채우기
            noise_real = torch.randn(C, self.H0, self.W0, dtype=img.dtype) * std_real
            noise_imag = torch.randn(C, self.H0, self.W0, dtype=img.dtype) * std_imag
            
            # 노이즈 텐서 생성 및 중앙에 크롭된 이미지 삽입
            padded_img = torch.stack([noise_real, noise_imag], dim=-1)
            paste_top, paste_left = pad_h // 2, pad_w // 2
            padded_img[:, paste_top:paste_top+Hc, paste_left:paste_left+Wc, :] = img_cropped
            
            final_img = padded_img
        else:
            # 패딩이 필요 없으면 크롭된 이미지를 그대로 사용
            final_img = img_cropped

        # 3. 다시 k-space로 변환
        k2 = torch_fft2c(final_img)

        # 4. 마스크도 동일하게 처리 (이 부분은 로직 변경 없음)
        w = mask.shape[-1] if mask.ndim > 1 else mask.shape[0]
        mw = mask if mask.ndim == 1 else mask.flatten()
        kept = min(w, self.W0)
        start = (w - kept) // 2
        m_c = mw[start:start+kept]
        pad_w_mask = self.W0 - kept
        pad_left = pad_w_mask // 2
        pad_right = pad_w_mask - pad_left
        m2 = F.pad(m_c, (pad_left, pad_right), mode='constant', value=0)

        attrs = dict(attrs)
        attrs['recon_size'] = [self.H0, self.W0]
        return m2, k2, target, attrs, fname, slice_idx

   
class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, mask, kspace, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1

        # mask = mask.byte().unsqueeze(-1)  # (H,W,1)
        # # mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        # return mask, kspace, target, maximum, fname, slice
    
        # kspace: complex dtype OR real/imag last‐dim 텐서
        # real/imag 형태가 아니면 complex → real/imag
        if not (kspace.ndim >=4 and kspace.shape[-1]==2):
            kspace = torch.view_as_real(to_tensor(kspace).to(torch.complex64))
        H = kspace.shape[-2]

        # print(mask.shape)
        # print(kspace.shape)

        mask = mask.reshape(1, 1, H, 1).float().byte()
        return mask, kspace, target, maximum, fname, slice

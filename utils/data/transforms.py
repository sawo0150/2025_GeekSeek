import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
from utils.data.transforms_Facebook import complex_center_crop
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
    (C, H, W) 복소 k-space를
      - H > H0: 중앙 H0 부분만 자르고
      - H < H0: 위아래 zero-pad
    같은 방식으로 W도 처리해서 항상 (C, H0, W0) 로 만듭니다.
    """
    def __init__(self, target_size: Tuple[int,int]):
        self.H0, self.W0 = target_size

    def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
        # kspace: torch.Tensor (C,H,W) complex represented as complex dtypece.real,kspace.imag),-1)
        # ❶ 입력을 complex64 dtype 으로 맞춤
        k_cplx = to_tensor(kspace).to(torch.complex64)

        # ❷ real/imag 채널로 변환
        k_ri = torch.view_as_real(k_cplx)   # (C, H, W, 2)

        # ❷ 이미지 도메인으로
        img = torch_ifft2c(k_ri)  # (C, H, W, 2)

        # ❸ crop or pad center in image domain
        C, H, W, _ = img.shape
        Hc, Wc = min(H,self.H0), min(W,self.W0)
        top, left = (H-Hc)//2, (W-Wc)//2
        img = img[..., top:top+Hc, left:left+Wc, :]
        pad_h, pad_w = self.H0-Hc, self.W0-Wc
        img = F.pad(img.permute(0,3,1,2), 
                    [pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2],
                    mode='constant', value=0
                   ).permute(0,2,3,1)

        # ❹ 다시 k-space로
        k2 = torch_fft2c(img)  # (C, H0, W0, 2)

        # ❺ mask도 동일하게 처리
        w = mask.shape[-1] if mask.ndim>1 else mask.shape[0]
        mw = mask if mask.ndim==1 else mask.flatten()
        kept = min(w, self.W0)
        start = (w - kept) // 2
        m_c = mw[start:start+kept]
        pad_w = self.W0 - kept
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
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

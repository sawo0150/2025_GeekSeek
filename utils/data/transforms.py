# utils/data/transforms.py

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
from collections.abc import Mapping
import fastmri

def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    return torch.as_tensor(data)

def to_realimag(x: torch.Tensor) -> torch.Tensor:
    if x.dtype.is_complex:
        return torch.view_as_real(x)
    if x.ndim >= 4 and x.shape[-1] == 2:
        return x
    raise ValueError(f"to_realimag: unexpected tensor shape/dtype {x.shape}, {x.dtype}")

def torch_ifft2c(x: torch.Tensor):
    xr = to_realimag(x)
    out = fastmri.ifft2c(xr)  
    return out

def torch_fft2c(x: torch.Tensor):
    xr = to_realimag(x)
    out = fastmri.fft2c(xr)
    return out

class MaskApplyTransform:
    def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
        k_cplx = to_tensor(kspace).to(torch.complex64)
        m = torch.from_numpy(mask).float()
        k = k_cplx * m
        return m, k, target, attrs, fname, slice_idx

# class CenterCropOrPad:
#     """
#     (C, H, W) 복소 k-space를 이미지 도메인에서 crop하거나,
#     배경 노이즈를 추정하여 noise padding을 수행합니다.
#     항상 (C, H0, W0) 크기를 반환합니다.
    
#     [수정] kspace뿐만 아니라 target 이미지도 동일하게 크롭/패딩합니다.
#     """
#     def __init__(self, target_size: Tuple[int,int], corner_size: int = 8):
#         self.H0, self.W0 = target_size
#         self.corner_size = corner_size
#     # ┕ [추가] 이미지 코너에서 노이즈 표준편차를 추정하는 헬퍼 메서드
#     def _estimate_noise_std(self, image_slice: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         이미지 네 귀퉁이에서 real/imag 채널의 노이즈 표준편차를 각각 계산합니다.
#         image_slice: (C, H, W, 2) 형태의 텐서
#         """
#         C, H, W, _ = image_slice.shape
#         # 코너 크기가 이미지 크기보다 크지 않도록 보정
#         cs = min(self.corner_size, H // 2, W // 2)
#         if cs == 0: # 이미지가 너무 작으면 표준편차 0으로 간주
#             return torch.tensor(0.0), torch.tensor(0.0)

#         # 네 귀퉁이 영역을 잘라 리스트에 추가
#         corners = [
#             image_slice[:, :cs, :cs, :],      # Top-left
#             image_slice[:, :cs, -cs:, :],     # Top-right
#             image_slice[:, -cs:, :cs, :],     # Bottom-left
#             image_slice[:, -cs:, -cs:, :],    # Bottom-right
#         ]
        
#         # 모든 코너를 하나의 텐서로 결합 (W 차원을 따라)
#         all_corners = torch.cat(corners, dim=2)
        
#         # real/imag 채널 각각의 표준편차 계산
#         std_real = torch.std(all_corners[..., 0])
#         std_imag = torch.std(all_corners[..., 1])
        
#         return std_real, std_imag

#     def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
#         # ── 0. attrs가 dict/Mapping 아니면 빈 dict로 대체 ─────────
#         if not isinstance(attrs, Mapping):
#             attrs = {}
        
#         # ── 1. K-SPACE 처리 (기존과 동일) ────────────────────────
#         k_cplx = to_tensor(kspace).to(torch.complex64)
#         k_ri = torch.view_as_real(k_cplx)
#         img = torch_ifft2c(k_ri)

#         # 현재 이 지점에서 ValueError가 발생하고 있습니다.
#         # img.shape가 (C, H, W, 2) 형태의 4차원이 아닐 때 발생하며,
#         # 이전 변환 단계에서 데이터의 차원이 예기치 않게 변경되었음을 의미합니다.
#         # 아래의 target 처리 로직을 포함하여 파이프라인 전체의 데이터 흐름을 안정화하면
#         # 이 문제가 해결될 가능성이 높습니다.
#         C, H, W, _ = img.shape
#         Hc, Wc = min(H, self.H0), min(W, self.W0)

#         top, left = (H - Hc) // 2, (W - Wc) // 2
#         img_cropped = img[:, top:top+Hc, left:left+Wc, :]

#         pad_h, pad_w = self.H0 - Hc, self.W0 - Wc
#         if pad_h > 0 or pad_w > 0:
#             std_real, std_imag = self._estimate_noise_std(img)
#             print("estimate_noise_std : ", std_real, std_imag)
#             noise_real = torch.randn(C, self.H0, self.W0, dtype=img.dtype) * std_real
#             noise_imag = torch.randn(C, self.H0, self.W0, dtype=img.dtype) * std_imag
#             padded_img = torch.stack([noise_real, noise_imag], dim=-1)
#             paste_top, paste_left = pad_h // 2, pad_w // 2
#             padded_img[:, paste_top:paste_top+Hc, paste_left:paste_left+Wc, :] = img_cropped
#             final_img = padded_img
#         else:
#             final_img = img_cropped

#         k2 = torch_fft2c(final_img)

#         # ── 3. MASK 처리 (기존과 동일) ─────────────────────────────
#         w = mask.shape[-1] if mask.ndim > 1 else mask.shape[0]
#         mw = mask if mask.ndim == 1 else mask.flatten()
#         kept = min(w, self.W0)
#         start = (w - kept) // 2
#         m_c = mw[start:start+kept]
#         pad_w_mask = self.W0 - kept
#         pad_left_mask = pad_w_mask // 2
#         pad_right_mask = pad_w_mask - pad_left_mask
#         m2 = F.pad(m_c, (pad_left_mask, pad_right_mask), mode='constant', value=0)

#         attrs = dict(attrs)
#         attrs['recon_size'] = [self.H0, self.W0]
        
#         # 수정된 kspace와 함께 수정된 target 반환
#         return m2, k2, target, attrs, fname, slice_idx

## 공통 유틸: mask crop/pad
def _crop_pad_mask(mask, W0):
    w = mask.shape[-1] if mask.ndim>1 else mask.shape[0]
    m = mask.flatten() if mask.ndim>1 else mask
    kept = min(w, W0)
    start = (w-kept)//2
    m_c = m[start:start+kept]
    pad = W0-kept
    l = pad//2; r = pad-l
    return F.pad(m_c, (l,r), mode='constant', value=0)
   
# -----------------------------------------------------------------------------
# 1) Image-space center crop + zero padding
class CenterCropZeroPad:
    def __init__(self, target_size: Tuple[int,int]):
        self.H0, self.W0 = target_size
        # print("CenterCropZeroPad!!!")
    def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
        if not isinstance(attrs, Mapping): attrs = {}
        k_cplx = to_tensor(kspace).to(torch.complex64)
        img = torch_ifft2c(torch.view_as_real(k_cplx))
        C, H, W, _ = img.shape
        Hc, Wc = min(H,self.H0), min(W,self.W0)
        top, left = (H-Hc)//2, (W-Wc)//2
        img = img[:, top:top+Hc, left:left+Wc, :]
        img = F.pad(img.permute(0,3,1,2),
                    [ (self.W0-Wc)//2, self.W0-Wc-(self.W0-Wc)//2,
                      (self.H0-Hc)//2, self.H0-Hc-(self.H0-Hc)//2 ],
                    mode='constant', value=0
                   ).permute(0,2,3,1)
        k2 = torch_fft2c(img)
        m2 = _crop_pad_mask(mask, self.W0)
        attrs = dict(attrs); attrs['recon_size']=[self.H0,self.W0]
        return m2, k2, target, attrs, fname, slice_idx

# -----------------------------------------------------------------------------
# 2) Image-space center crop + noise padding
class CenterCropNoisePad(CenterCropZeroPad):
    def __init__(self, target_size: Tuple[int,int], corner_size: int = 8):
        super().__init__(target_size)
        self.corner_size = corner_size
        # print("CenterCropNoisePad!!!")
    def _estimate_noise_std(self, image_slice):
        """
        이미지 네 귀퉁이에서 real/imag 채널의 노이즈 표준편차를 각각 계산합니다.
        image_slice: (C, H, W, 2) 형태의 텐서
        """
        C,H,W,_=image_slice.shape
        cs=min(self.corner_size,H//2,W//2)
        if cs==0: return torch.tensor(0.), torch.tensor(0.)
        corners=[
            image_slice[:,:cs,:cs,:], image_slice[:,:cs,-cs:,:],
            image_slice[:,-cs:,:cs,:], image_slice[:,-cs:,-cs:,:],
        ]
        allc=torch.cat(corners,dim=2)
        return torch.std(allc[...,0]), torch.std(allc[...,1])
    def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
        if not isinstance(attrs, Mapping): attrs = {}
        k_cplx = to_tensor(kspace).to(torch.complex64)
        img = torch_ifft2c(torch.view_as_real(k_cplx))
        C, H, W, _ = img.shape
        Hc, Wc = min(H,self.H0), min(W,self.W0)
        top, left = (H-Hc)//2, (W-Wc)//2
        img_crop = img[:, top:top+Hc, left:left+Wc, :]
        pad_h, pad_w = self.H0-Hc, self.W0-Wc
        if pad_h>0 or pad_w>0:
            sr, si = self._estimate_noise_std(img)
            noise = torch.randn(C,self.H0,self.W0,2,dtype=img.dtype)
            noise[...,0]*=sr; noise[...,1]*=si
            img = noise
            img[:, pad_h//2:pad_h//2+Hc, pad_w//2:pad_w//2+Wc, :] = img_crop
        else:
            img = img_crop
        k2 = torch_fft2c(img)
        m2 = _crop_pad_mask(mask, self.W0)
        attrs = dict(attrs); attrs['recon_size']=[self.H0,self.W0]
        return m2, k2, target, attrs, fname, slice_idx

# -----------------------------------------------------------------------------
# 3) k-space–domain center crop/pad (zero pad)
class KspaceCenterCropPad:
    def __init__(self, target_size: Tuple[int,int]):
        self.H0, self.W0 = target_size
        # print("KspaceCenterCropPad!!!")
    # def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
    #     if not isinstance(attrs, Mapping): attrs = {}
    #     k = to_tensor(kspace).to(torch.complex64)
    #     C, H, W = k.shape
    #     Hc, Wc = min(H,self.H0), min(W,self.W0)
    #     top, left = (H-Hc)//2, (W-Wc)//2
    #     k_crop = k[:, top:top+Hc, left:left+Wc]
    #     pad_h, pad_w = self.H0-Hc, self.W0-Wc
    #     if pad_h>0:
    #         pad = torch.zeros(C,pad_h,W, dtype=k.dtype, device=k.device)
    #         k = torch.cat([pad, k_crop, pad], dim=1)
    #     else:
    #         k = k_crop
    #     if pad_w>0:
    #         pad = torch.zeros(C,self.H0,pad_w, dtype=k.dtype, device=k.device)
    #         leftw, rightw = pad_w//2, pad_w-pad_w//2
    #         k = torch.cat([pad[:,:,:leftw], k, pad[:,:,leftw:]], dim=2)
    #     attrs = dict(attrs); attrs['recon_size']=[self.H0,self.W0]
    #     # mask도 같은 로직
    #     m2 = _crop_pad_mask(mask, self.W0)
    #     return m2, k, target, attrs, fname, slice_idx

    def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
        """
        k-space 도메인에서 center crop & zero-pad.
        """
        if not isinstance(attrs, Mapping):
            attrs = {}
        # 1) tensor 변환
        k = to_tensor(kspace).to(torch.complex64)  # (C, H, W)

        # 2) center crop
        C, H, W = k.shape
        Hc, Wc = min(H, self.H0), min(W, self.W0)
        top, left = (H - Hc) // 2, (W - Wc) // 2
        k_crop = k[:, top:top+Hc, left:left+Wc]

        # 3) pad 계산
        pad_h = self.H0 - Hc
        pad_w = self.W0 - Wc
        pad_top    = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left   = pad_w // 2
        pad_right  = pad_w - pad_left

        # 4) F.pad 로 한 번에 처리 (pad = (w_l, w_r, h_t, h_b))
        k2 = F.pad(
            k_crop,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=0
        )  # 결과 (C, H0, W0)

        # 5) mask도 동일하게 center-crop+pad
        m2 = _crop_pad_mask(mask, self.W0)

        # 6) 리턴
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
    
        if not (kspace.ndim >=4 and kspace.shape[-1]==2):
            kspace = torch.view_as_real(to_tensor(kspace).to(torch.complex64))

        # ✨ [최종 수정] VarNet 모델이 기대하는 5차원 텐서가 되도록 4차원 형태로 마스크 reshape
        w = mask.shape[0]
        # (너비,) -> (1, 1, 너비, 1) 형태로 변경 -> DataLoader 거치면 (B, 1, 1, W, 1)이 됨
        mask = mask.reshape(1, 1, w, 1).float().byte()
        
        return mask, kspace, target, maximum, fname, slice
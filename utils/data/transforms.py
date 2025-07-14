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

class CenterCropOrPad:
    def __init__(self, target_size: Tuple[int,int], use_noise_padding: bool = False, corner_size: int = 16):
        self.H0, self.W0 = target_size
        self.use_noise_padding = use_noise_padding
        self.corner_size = corner_size

    def _estimate_noise_std(self, image_slice: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        C, H, W, _ = image_slice.shape
        cs = min(self.corner_size, H // 2, W // 2)
        if cs == 0:
            return torch.tensor(0.0), torch.tensor(0.0)

        corners = [
            image_slice[:, :cs, :cs, :], image_slice[:, :cs, -cs:, :],
            image_slice[:, -cs:, :cs, :], image_slice[:, -cs:, -cs:, :],
        ]
        all_corners = torch.cat(corners, dim=2)
        std_real = torch.std(all_corners[..., 0])
        std_imag = torch.std(all_corners[..., 1])
        return std_real, std_imag

    def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
        if not isinstance(attrs, Mapping):
            attrs = {}
        
        k_cplx = to_tensor(kspace).to(torch.complex64)
        k_ri = torch.view_as_real(k_cplx)
        img = torch_ifft2c(k_ri)

        C, H, W, _ = img.shape
        Hc, Wc = min(H, self.H0), min(W, self.W0)

        top, left = (H - Hc) // 2, (W - Wc) // 2
        img_cropped = img[:, top:top+Hc, left:left+Wc, :]

        pad_h, pad_w = self.H0 - Hc, self.W0 - Wc

        if pad_h > 0 or pad_w > 0:
            if self.use_noise_padding:
                std_real, std_imag = self._estimate_noise_std(img)
                noise_real = torch.randn(C, self.H0, self.W0, dtype=img.dtype, device=img.device) * std_real
                noise_imag = torch.randn(C, self.H0, self.W0, dtype=img.dtype, device=img.device) * std_imag
                padded_img = torch.stack([noise_real, noise_imag], dim=-1)
                paste_top, paste_left = pad_h // 2, pad_w // 2
                padded_img[:, paste_top:paste_top+Hc, paste_left:left+Wc, :] = img_cropped
                final_img = padded_img
            else:
                pad_h_top = pad_h // 2
                pad_h_bot = pad_h - pad_h_top
                pad_w_left = pad_w // 2
                pad_w_right = pad_w - pad_w_left
                pad_dims = (pad_w_left, pad_w_right, pad_h_top, pad_h_bot)
                permuted_cropped = img_cropped.permute(0, 3, 1, 2)
                padded_permuted = F.pad(permuted_cropped, pad_dims, "constant", 0)
                final_img = padded_permuted.permute(0, 2, 3, 1)
        else:
            final_img = img_cropped

        k2 = torch_fft2c(final_img)

        final_target = target
        if isinstance(target, (np.ndarray, torch.Tensor)):
            target_tensor = to_tensor(target).float()
            if target_tensor.ndim == 2:
                h_t, w_t = target_tensor.shape
                top_t, left_t = (h_t - Hc) // 2, (w_t - Wc) // 2
                target_cropped = target_tensor[top_t:top_t+Hc, left_t:left_t+Wc]

                if pad_h > 0 or pad_w > 0:
                    pad_left_t, pad_right_t = pad_w // 2, pad_w - (pad_w // 2)
                    pad_top_t, pad_bot_t = pad_h // 2, pad_h - (pad_h // 2)
                    final_target = F.pad(target_cropped, (pad_left_t, pad_right_t, pad_top_t, pad_bot_t), "constant", 0)
                else:
                    final_target = target_cropped

        w_mask = mask.shape[-1] if mask.ndim > 1 else mask.shape[0]
        mw = mask if mask.ndim == 1 else mask.flatten()
        kept = min(w_mask, self.W0)
        start = (w_mask - kept) // 2
        m_c = mw[start:start+kept]
        pad_w_mask = self.W0 - kept
        pad_left_mask = pad_w_mask // 2
        pad_right_mask = pad_w_mask - pad_left_mask
        m2 = F.pad(m_c, (pad_left_mask, pad_right_mask), mode='constant', value=0)

        attrs = dict(attrs)
        attrs['recon_size'] = [self.H0, self.W0]
        
        return m2, k2, final_target, attrs, fname, slice_idx

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
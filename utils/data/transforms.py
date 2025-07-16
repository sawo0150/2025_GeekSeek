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
    if x.ndim >= 3 and x.shape[-1] == 2:
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
        if isinstance(kspace, torch.Tensor) and kspace.shape[-1] == 2:
            k_cplx = torch.view_as_complex(kspace)
        else:  # numpy array
            k_cplx = to_tensor(kspace).to(torch.complex64)

        m_tensor = torch.from_numpy(mask).float()
        k_masked = k_cplx * m_tensor.reshape(1, 1, -1)

        # 다음 DataTransform이 real/imag 텐서를 기대하므로 타입을 맞춰서 반환
        return mask, torch.view_as_real(k_masked), target, attrs, fname, slice_idx

class ImageSpaceCropTransform:
    def __init__(self, target_size: Tuple[int, int], corner_size: int = 16):
        self.H0, self.W0 = target_size
        self.corner_size = corner_size
        
    def _estimate_noise_std(self, image_slice: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        C, H, W, _ = image_slice.shape
        cs = min(self.corner_size, H // 2, W // 2)
        if cs == 0:
            return torch.tensor(0.0), torch.tensor(0.0)
        first_coil_img = image_slice[0, ...]
        corners = [
            first_coil_img[:cs, :cs, :], first_coil_img[:cs, -cs:, :],
            first_coil_img[-cs:, :cs, :], first_coil_img[-cs:, -cs:, :],
        ]
        all_corners = torch.cat([c.reshape(-1, 2) for c in corners], dim=0)
        return torch.std(all_corners[:, 0]), torch.std(all_corners[:, 1])

    def _crop_and_pad(self, data_tensor: torch.Tensor, is_kspace: bool) -> torch.Tensor:
        if is_kspace:
            img = torch_ifft2c(data_tensor)
        else:
            img = data_tensor
        C, H, W, _ = img.shape
        Hc, Wc = min(H, self.H0), min(W, self.W0)
        top, left = (H - Hc) // 2, (W - Wc) // 2
        img_cropped = img[:, top:top+Hc, left:left+Wc, :]
        pad_h, pad_w = self.H0 - Hc, self.W0 - Wc
        if pad_h > 0 or pad_w > 0:
            if is_kspace:
                std_real, std_imag = self._estimate_noise_std(img)
                noise_real = torch.randn(C, self.H0, self.W0, dtype=img.dtype, device=img.device) * std_real
                noise_imag = torch.randn(C, self.H0, self.W0, dtype=img.dtype, device=img.device) * std_imag
                padded_img = torch.stack([noise_real, noise_imag], dim=-1)
            else:
                padded_img = torch.zeros(C, self.H0, self.W0, 2, dtype=img.dtype, device=img.device)
            paste_top, paste_left = pad_h // 2, pad_w // 2
            padded_img[:, paste_top:paste_top+Hc, paste_left:paste_left+Wc, :] = img_cropped
            final_img = padded_img
        else:
            final_img = img_cropped
        return torch_fft2c(final_img) if is_kspace else final_img

    def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
        attrs_copy = dict(attrs)

        if isinstance(kspace, torch.Tensor) and kspace.shape[-1] == 2:
            k_cplx = torch.view_as_complex(kspace)
        else:
            k_cplx = to_tensor(kspace).to(torch.complex64)
        
        k_ri = torch.view_as_real(k_cplx)
        k_processed = self._crop_and_pad(k_ri, is_kspace=True)

        if isinstance(target, np.ndarray) and target.ndim == 2:
            target_tensor = to_tensor(target).float()
            target_ri = torch.stack([target_tensor, torch.zeros_like(target_tensor)], dim=-1).unsqueeze(0)
            target_processed_ri = self._crop_and_pad(target_ri, is_kspace=False)
            target_processed = target_processed_ri[0, ..., 0].numpy()
        else:
            target_processed = target

        w = mask.shape[0]
        kept = min(w, self.W0)
        start = (w - kept) // 2
        m_c = mask[start:start+kept]
        pad_w_mask = self.W0 - kept
        
        mask_processed = np.pad(
            m_c, 
            (pad_w_mask//2, pad_w_mask - pad_w_mask//2), 
            mode='constant', 
            constant_values=0
        )
        
        attrs_copy['recon_size'] = [self.H0, self.W0]
        
        k_processed_cplx = torch.view_as_complex(k_processed).numpy()

        return mask_processed, k_processed_cplx, target_processed, attrs_copy, fname, slice_idx

class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key

    def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
        if not self.isforward:
            target = to_tensor(target)
            # HDF5 속성에서 maximum 가져오기, 없거나 0에 가까우면 target.max() 사용
            maximum = attrs.get(self.max_key, None)
            if maximum is None or float(maximum) <= 1e-10:
                maximum = float(target.max())
                # print(f"경고: '{self.max_key}'가 {fname}에 없거나 너무 작습니다. target.max()={maximum:.4e} 사용")
            else:
                maximum = float(maximum)
            maximum = to_tensor(maximum)
        else:
            target = torch.tensor(-1.0)
            maximum = torch.tensor(-1.0)

        kspace_ri = to_tensor(kspace)

        w = mask.shape[0]
        mask_tensor = torch.from_numpy(mask).reshape(1, 1, w, 1).float().byte()

        return mask_tensor, kspace_ri, target, maximum, fname, slice_idx
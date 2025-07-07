"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Union, Mapping

try:
    from pytorch_msssim import ms_ssim
    _MS_SSIM_AVAILABLE = True
except ImportError:
    _MS_SSIM_AVAILABLE = False

def _get_threshold(mask_threshold: Union[float, Mapping[str, float]], cat: str) -> float:
    """
    Retrieve threshold based on cat if mask_threshold is a mapping,
    otherwise return constant float.
    """
    if mask_threshold is None:
        raise ValueError("mask_threshold is None")
    if isinstance(mask_threshold, Mapping):
        if cat not in mask_threshold:
            raise KeyError(f"Category '{cat}' not found in mask_threshold mapping")
        return float(mask_threshold[cat])
    else:
        return float(mask_threshold)


class MaskedLoss(nn.Module):
    """
    Base class for losses with optional masking and region weighting.

    Args:
        mask_threshold float or Mapping[str, float] or None : if set, threshold for mask creation based on target.
        mask_only (bool): if True, apply loss only within the mask region.
        region_weight (bool): if True, multiply loss by (mask_area / total_area).
    """
    def __init__(self,
                 mask_threshold: Union[float, Mapping[str, float]] = None,
                 mask_only: bool = False,
                 region_weight: bool = False):
        super().__init__()
        self.mask_threshold = mask_threshold
        self.mask_only      = mask_only
        self.region_weight  = region_weight

    def forward(
        self,
        output: torch.Tensor,         # [B, H, W] or [H, W]
        target: torch.Tensor,         # [B, H, W] or [H, W]
        data_range=None,
        cats: Union[str, List[str]] = None,
        ) -> torch.Tensor:
        # output, target: [B, H, W] or [H, W]
        # 1) build mask if needed
        mask = None
        if self.mask_threshold is not None:
            if cats is None:
                raise ValueError("`cats` 리스트를 반드시 전달해야 합니다.")
            # 단일 샘플인 경우에도 리스트로
            if not isinstance(cats, list):
                cats = [cats]
                # [H,W] → [1,H,W]
                if target.dim() == 2:
                    target = target.unsqueeze(0)
                    output = output.unsqueeze(0)

            mask = self._make_mask(target, cats)

        # 2) apply mask_only: zero-out outside-mask
        if mask is not None and self.mask_only:
            output = output * mask
            target = target * mask

        # 3) compute base loss
        loss = self.compute_loss(output, target, data_range)

        # 4) apply region_weight: scale by mask coverage
        if mask is not None and self.region_weight:
            w = mask.sum(dim=[1,2]) / (mask.shape[1]*mask.shape[2])  # [B]
            # 배치별로 적용하려면
            loss = (loss.view(-1) * w).mean()  # compute_loss must return per-sample loss tensor

        return loss

    def compute_loss(self,
                     output: torch.Tensor,
                     target: torch.Tensor,
                     data_range) -> torch.Tensor:
        raise NotImplementedError

    def _make_mask(
        self,
        targets: torch.Tensor,                 # [B, H, W]
        cats: List[str],
    ) -> torch.Tensor:
        """
        각 배치 i 에 대해 cats[i]에 대응하는 threshold 로
        형태학적 처리된 mask 생성 → [B, H, W]
        """
        device = targets.device
        t_np   = targets.detach().cpu().numpy()  # (B,H,W)
        masks  = []
        kernel = np.ones((3,3), np.uint8)

        for i, cat in enumerate(cats):
            thr = _get_threshold(self.mask_threshold, cat)
            m   = (t_np[i] > thr).astype(np.uint8)
            m   = cv2.erode(m,   kernel, iterations=1)
            m   = cv2.dilate(m,  kernel, iterations=15)
            m   = cv2.erode(m,   kernel, iterations=14)
            masks.append(m)

        mask_np = np.stack(masks, axis=0)       # (B,H,W)
        return torch.from_numpy(mask_np).to(device).float()

# class SSIMLoss(nn.Module):
#     """
#     SSIM loss module.
#     """

#     def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
#         """
#         Args:
#             win_size: Window size for SSIM calculation.
#             k1: k1 parameter for SSIM calculation.
#             k2: k2 parameter for SSIM calculation.
#         """
#         super().__init__()
#         self.win_size = win_size
#         self.k1, self.k2 = k1, k2
#         self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
#         NP = win_size ** 2
#         self.cov_norm = NP / (NP - 1)

#     def forward(self, X, Y, data_range):
#         X = X.unsqueeze(1)
#         Y = Y.unsqueeze(1)
#         data_range = data_range[:, None, None, None]
#         C1 = (self.k1 * data_range) ** 2
#         C2 = (self.k2 * data_range) ** 2
#         ux = F.conv2d(X, self.w)
#         uy = F.conv2d(Y, self.w)
#         uxx = F.conv2d(X * X, self.w)
#         uyy = F.conv2d(Y * Y, self.w)
#         uxy = F.conv2d(X * Y, self.w)
#         vx = self.cov_norm * (uxx - ux * ux)
#         vy = self.cov_norm * (uyy - uy * uy)
#         vxy = self.cov_norm * (uxy - ux * uy)
#         A1, A2, B1, B2 = (
#             2 * ux * uy + C1,
#             2 * vxy + C2,
#             ux ** 2 + uy ** 2 + C1,
#             vx + vy + C2,
#         )
#         D = B1 * B2
#         S = (A1 * A2) / D

#         return 1 - S.mean()


class L1LossWrapper(nn.Module):
    """train_part.train_epoch가 maximum을 넘겨도 무시하도록 3-인자 래퍼"""
    def __init__(self): super().__init__(); self.loss = nn.L1Loss()
    def forward(self, output, target, maximum=None):
        return self.loss(output, target)
    

class L1Loss(MaskedLoss):
    """L1 loss with optional masking and region weighting."""
    def __init__(self,
                 mask_threshold: Union[float, Mapping[str, float]] = None,
                 mask_only: bool = False,
                 region_weight: bool = False):
        super().__init__(mask_threshold, mask_only, region_weight)

    def compute_loss(self, output, target, data_range):
        return F.l1_loss(output, target)

class SSIMLoss(MaskedLoss):
    """
    SSIM loss (1 - SSIM) with optional masking and weighting.
    """
    def __init__(
        self,
        win_size: int = 7,
        k1: float = 0.01,
        k2: float = 0.03,
        mask_threshold: Union[float, Mapping[str, float]] = None,
        mask_only: bool = False,
        region_weight: bool = False):

        super().__init__(mask_threshold, mask_only, region_weight)
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer('w', torch.ones(1, 1, win_size, win_size) / (win_size ** 2))
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)


    def compute_loss(self, output, target, data_range):
        # output, target: [B, H, W] or [H, W]
        # ensure dims: [B, 1, H, W]
        if output.ndim == 3:
            X = output.unsqueeze(1)
            Y = target.unsqueeze(1)
        elif output.ndim == 2:
            X = output.unsqueeze(0).unsqueeze(0)
            Y = target.unsqueeze(0).unsqueeze(0)
        else:
            raise NotImplementedError(f"SSIM expects 2D or 3D tensor, got {output.shape}")

        d = float(data_range) if data_range is not None else 1.0
        C1 = (self.k1 * d) ** 2
        C2 = (self.k2 * d) ** 2
        ux  = F.conv2d(X, self.w)
        uy  = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx  = self.cov_norm * (uxx - ux * ux)
        vy  = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1 = 2 * ux * uy + C1
        A2 = 2 * vxy + C2
        B1 = ux * ux + uy * uy + C1
        B2 = vx + vy + C2
        S = (A1 * A2) / (B1 * B2)
        return 1.0 - S.mean()


class MSSSIMLoss(MaskedLoss):
    """
    MS-SSIM loss (1 - MS-SSIM) with optional masking.
    Requires pytorch_msssim.
    """
    def __init__(self,
                 data_range: float = 1.0,
                 size_average: bool = True,
                 mask_threshold: Union[float, Mapping[str, float]] = None,
                 mask_only: bool = False,
                 region_weight: bool = False):
        
        super().__init__(mask_threshold, mask_only, region_weight)
        if not _MS_SSIM_AVAILABLE:
            raise ImportError("pytorch_msssim is required for MSSSIMLoss")
        self.data_range = data_range
        self.size_average = size_average

    def compute_loss(self, output, target, data_range):
        if output.ndim == 3:
            X = output.unsqueeze(1)
            Y = target.unsqueeze(1)
        elif output.ndim == 2:
            X = output.unsqueeze(0).unsqueeze(0)
            Y = target.unsqueeze(0).unsqueeze(0)
        else:
            raise NotImplementedError
        return 1.0 - ms_ssim(
            X, Y,
            data_range=self.data_range,
            size_average=self.size_average,
        )



class PSNRLoss(MaskedLoss):
    """
    PSNR-based loss (-PSNR) with optional masking.
    """
    def __init__(
        self,
        data_range: float = 1.0,
        mask_threshold: Union[float, Mapping[str, float]] = None,
        mask_only: bool = False,
        region_weight: bool = False):
        
        super().__init__(mask_threshold, mask_only, region_weight)
        self.data_range = data_range

    def compute_loss(self, output, target, data_range):
        mse = F.mse_loss(output, target)
        psnr = 10 * torch.log10(self.data_range ** 2 / (mse + 1e-12))
        return -psnr


class SSIML1Loss(MaskedLoss):
    """Combined SSIM + L1 loss with optional masking and weighting."""
    def __init__(self,
                 win_size: int = 7,
                 k1: float = 0.01,
                 k2: float = 0.03,
                 weight_ssim: float = 1.0,
                 weight_l1: float = 1.0,
                 mask_threshold: Union[float, Mapping[str, float]] = None,
                 mask_only: bool = False,
                 region_weight: bool = False):
        super().__init__(mask_threshold, mask_only, region_weight)
        self.ssim_base = SSIMLoss(win_size, k1, k2)
        self.weight_ssim = weight_ssim
        self.weight_l1 = weight_l1

    def compute_loss(self, output, target, data_range):
        ssim_loss = self.ssim_base.compute_loss(output, target, data_range)
        l1_loss = F.l1_loss(output, target)
        return self.weight_ssim * ssim_loss + self.weight_l1 * l1_loss


class MSSSIML1Loss(MaskedLoss):
    """Combined MS-SSIM + L1 loss with optional masking and weighting."""
    def __init__(self,
                 data_range: float = 1.0,
                 size_average: bool = True,
                 weight_ms_ssim: float = 1.0,
                 weight_l1: float = 1.0,
                 mask_threshold: Union[float, Mapping[str, float]] = None,
                 mask_only: bool = False,
                 region_weight: bool = False):
        super().__init__(mask_threshold, mask_only, region_weight)
        if not _MS_SSIM_AVAILABLE:
            raise ImportError("pytorch_msssim is required for MSSSIML1Loss")
        self.data_range = data_range
        self.size_average = size_average
        self.weight_ms_ssim = weight_ms_ssim
        self.weight_l1 = weight_l1

    def compute_loss(self, output, target, data_range):
        # MS-SSIM
        if output.ndim == 3:
            X = output.unsqueeze(1)
            Y = target.unsqueeze(1)
        elif output.ndim == 2:
            X = output.unsqueeze(0).unsqueeze(0)
            Y = target.unsqueeze(0).unsqueeze(0)
        else:
            raise NotImplementedError
        ms_loss = 1.0 - ms_ssim(X, Y, data_range=self.data_range, size_average=self.size_average)
        l1_loss = F.l1_loss(output, target)
        return self.weight_ms_ssim * ms_loss + self.weight_l1 * l1_loss

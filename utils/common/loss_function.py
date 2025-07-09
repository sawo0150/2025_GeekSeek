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
    from pytorch_msssim import MS_SSIM, ms_ssim
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

            # 목록 또는 튜플로 통일
            cats_list = cats if isinstance(cats, (list, tuple)) else [cats]
            # 튜플/리스트 원소는 첫 번째 항목(카테고리명)만 추출
            cats_list = [
                c[0] if isinstance(c, (tuple, list)) and len(c) > 0 else c
                for c in cats_list
            ]
            # [H,W]일 때는 배치 차원 추가
            if output.dim() == 2:
                output = output.unsqueeze(0)
                target = target.unsqueeze(0)
            mask = self._make_mask(target, cats_list)

        # 2) apply mask_only: zero-out outside-mask
        if mask is not None and self.mask_only:
            output = output * mask
            target = target * mask

        # 3) compute base loss
        loss = self.compute_loss(output, target, data_range)

        # # 4) apply region_weight: scale by mask coverage
        # if mask is not None and self.region_weight:
        #     w = mask.sum(dim=[1,2]) / (mask.shape[1]*mask.shape[2])  # [B]
        #     # 배치별로 적용하려면
        #     loss = (loss.view(-1) * w).mean()  # compute_loss must return per-sample loss tensor

    
        # 4) apply region_weight: scale each sample's loss by mask coverage (per-sample 벡터 유지)
        if mask is not None and self.region_weight:
            # mask.sum → [B], 영역 비율
            w = mask.sum(dim=[1,2]) / (mask.shape[1] * mask.shape[2])
            # per-sample loss 벡터에 가중치 곱하기
            loss = loss.view(-1) * w

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


# class L1LossWrapper(nn.Module):
#     """train_part.train_epoch가 maximum을 넘겨도 무시하도록 3-인자 래퍼"""
#     def __init__(self): super().__init__(); self.loss = nn.L1Loss()
#     def forward(self, output, target, maximum=None):
        # return self.loss(output, target)
    

class L1Loss(MaskedLoss):
    """L1 loss with optional masking and region weighting."""
    def __init__(self,
                 mask_threshold: Union[float, Mapping[str, float]] = None,
                 mask_only: bool = False,
                 region_weight: bool = False):
        super().__init__(mask_threshold, mask_only, region_weight)
    
    def compute_loss(self, output, target, data_range):
        # 1) data_range 반영 (스칼라/텐서 모두 지원)
        if data_range is None:
            d = 1.0
        elif isinstance(data_range, torch.Tensor):
            d = data_range.view(-1, *([1] * (output.dim() - 1)))
        else:
            d = float(data_range)

        # 2) 픽셀별 L1 차이 계산
        per_pixel = F.l1_loss(output, target, reduction='none') / d

        # 3) 배치별 평균 → [B] 크기의 텐서 반환
        return per_pixel.view(per_pixel.size(0), -1).mean(dim=1)


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

        # print(data_range)
        # d = float(data_range) if data_range is not None else 1.0
        # data_range 반영 (스칼라 혹은 배치 텐서)
        if data_range is None:
            d = 1.0
        elif isinstance(data_range, torch.Tensor):
            d = data_range.view(-1,1,1,1)
        else:
            d = float(data_range)

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
        # return 1.0 - S.mean()
        # per-sample SSIM loss 반환
        return (1.0 - S).view(S.size(0), -1).mean(dim=1)

class MSSSIMLoss(MaskedLoss):
    """
    MS-SSIM loss (1 - MS-SSIM) with optional masking.
    Uses functional ms_ssim for dynamic data_range.
    """
    def __init__(self,
                 data_range: float = 1.0,
                 size_average: bool = False,   # per-sample loss를 원하면 False
                 mask_threshold=None,
                 mask_only=False,
                 region_weight=False):
        super().__init__(mask_threshold, mask_only, region_weight)
        if not _MS_SSIM_AVAILABLE:
            raise ImportError("pytorch_msssim이 필요합니다.")
        self.data_range   = data_range
        self.size_average = size_average

    def compute_loss(self, output, target, data_range):
        # 1) [B,H,W] or [H,W] → [B,1,H,W]
        if output.dim() == 3:
            X, Y = output.unsqueeze(1), target.unsqueeze(1)
        elif output.dim() == 2:
            X = output.unsqueeze(0).unsqueeze(0)
            Y = target.unsqueeze(0).unsqueeze(0)
        else:
            raise NotImplementedError(f"Unsupported shape {output.shape}")

        # 2) 동적 data_range 반영
        dr = data_range if data_range is not None else self.data_range
        if isinstance(dr, torch.Tensor):
            dr = float(dr.view(-1)[0])

        # 3) 필요하다면 패딩 (멀티스케일 크기 불일치 방지)
        #    기본 levels=5 → 2^(5-1)=16 배수로 맞추기
        B, C, H, W = X.shape
        pad_H = (16 - H % 16) % 16
        pad_W = (16 - W % 16) % 16
        if pad_H or pad_W:
            X = F.pad(X, (0, pad_W, 0, pad_H), mode='reflect')
            Y = F.pad(Y, (0, pad_W, 0, pad_H), mode='reflect')

        # 4) ms_ssim 함수로 계산
        loss = 1.0 - ms_ssim(
            X, Y,
            data_range=dr,
            size_average=self.size_average,
            win_size=11  # 기본 윈도우 크기
        )

        # 5) 스칼라가 나온 경우 대비 → 항상 (B,) 형태로
        if loss.dim() == 0:
            loss = loss.unsqueeze(0).expand(B)

        return loss


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

    # def compute_loss(self, output, target, data_range):
    #     mse = F.mse_loss(output, target)
    #     psnr = 10 * torch.log10(self.data_range ** 2 / (mse + 1e-12))
    #     return -psnr
    
    def compute_loss(self, output, target, data_range):
        # MSE per-sample 계산
        mse = F.mse_loss(output, target, reduction='none')
        mse = mse.view(mse.size(0), -1).mean(dim=1)
        # data_range는 인스턴스 변수 혹은 파라미터 우선
        dr = self.data_range if data_range is None else (data_range if isinstance(data_range, torch.Tensor) else float(data_range))
        psnr = 10 * torch.log10(dr ** 2 / (mse + 1e-12))
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

    # def compute_loss(self, output, target, data_range):
    #     # 1) SSIM part (unchanged)
    #     ssim_loss = self.ssim_base.compute_loss(output, target, data_range)
    #     # 2) L1 part: data_range 로 정규화
    #     d = float(data_range) if data_range is not None else 1.0
    #     l1_loss = F.l1_loss(output, target) / d
    #     return self.weight_ssim * ssim_loss + self.weight_l1 * l1_loss

    def compute_loss(self, output, target, data_range):
        # 1) SSIM part (per-sample)
        ssim_loss = self.ssim_base.compute_loss(output, target, data_range)
        # 2) L1 part: 스칼라/텐서 data_range 모두 지원
        if data_range is None:
            d = 1.0
        elif isinstance(data_range, torch.Tensor):
            d = data_range.view(-1, *([1] * (output.dim() - 1)))
        else:
            d = float(data_range)
        l1 = F.l1_loss(output, target, reduction='none') / d
        l1_loss = l1.view(l1.size(0), -1).mean(dim=1)
        return self.weight_ssim * ssim_loss + self.weight_l1 * l1_loss

# class MSSSIML1Loss(MaskedLoss):
#     """Combined MS-SSIM + L1 loss with optional masking and weighting."""
#     def __init__(self,
#                  data_range: float = 1.0,
#                  size_average: bool = True,
#                  weight_ms_ssim: float = 1.0,
#                  weight_l1: float = 1.0,
#                  mask_threshold: Union[float, Mapping[str, float]] = None,
#                  mask_only: bool = False,
#                  region_weight: bool = False):
#         super().__init__(mask_threshold, mask_only, region_weight)
#         if not _MS_SSIM_AVAILABLE:
#             raise ImportError("pytorch_msssim is required for MSSSIML1Loss")
#         self.data_range = data_range
#         self.size_average = size_average
#         self.weight_ms_ssim = weight_ms_ssim
#         self.weight_l1 = weight_l1

#     def compute_loss(self, output, target, data_range):
#         # 1) MS-SSIM part
#         if output.ndim == 3:
#             X = output.unsqueeze(1);  Y = target.unsqueeze(1)
#         elif output.ndim == 2:
#             X = output.unsqueeze(0).unsqueeze(0)
#             Y = target.unsqueeze(0).unsqueeze(0)
#         else:
#             raise NotImplementedError
#         # dr = float(data_range) if data_range is not None else self.data_range
#         # ms_loss = 1.0 - ms_ssim(X, Y, data_range=dr, size_average=self.size_average)

#         # data_range 반영
#         if data_range is None:
#             dr = self.data_range
#         elif isinstance(data_range, torch.Tensor):
#             dr = data_range
#         else:
#             dr = float(data_range)
#         ms_loss = 1.0 - ms_ssim(X, Y, data_range=dr, size_average=self.size_average)
#         ms_loss = ms_loss.view(ms_loss.size(0))

#         # 2) L1 part: data_range 로 정규화
#         # l1_loss = F.l1_loss(output, target) / dr
#         # return self.weight_ms_ssim * ms_loss + self.weight_l1 * l1_loss
    
#         # L1 part
#         l1 = F.l1_loss(output, target, reduction='none') / dr
#         l1_loss = l1.view(l1.size(0), -1).mean(dim=1)
#         return self.weight_ms_ssim * ms_loss + self.weight_l1 * l1_loss

class MSSSIML1Loss(MaskedLoss):
    """Combined MS-SSIM + L1 loss with optional masking and weighting."""
    def __init__(self,
                 data_range: float = 1.0,
                 size_average: bool = False,
                 weight_ms_ssim: float = 1.0,
                 weight_l1: float = 1.0,
                 mask_threshold: Union[float, Mapping[str, float]] = None,
                 mask_only: bool = False,
                 region_weight: bool = False):
        super().__init__(mask_threshold, mask_only, region_weight)
        if not _MS_SSIM_AVAILABLE:
            raise ImportError("pytorch_msssim이 필요합니다.")
        # base MS-SSIM loss
        self.msssim_base = MSSSIMLoss(
            data_range=data_range,
            size_average=size_average,
            mask_threshold=mask_threshold,
            mask_only=mask_only,
            region_weight=region_weight
        )
        self.weight_ms_ssim = weight_ms_ssim
        self.weight_l1 = weight_l1

    def compute_loss(self, output, target, data_range):
        # 1) MS-SSIM part via base class
        ms_loss = self.msssim_base.compute_loss(output, target, data_range)

        # 2) L1 part: normalize by data_range
        if data_range is None:
            dr = 1.0
        elif isinstance(data_range, torch.Tensor):
            dr = data_range.view(-1, *([1] * (output.dim() - 1)))
        else:
            dr = float(data_range)
        per_pixel = F.l1_loss(output, target, reduction='none') / dr
        l1_loss = per_pixel.view(per_pixel.size(0), -1).mean(dim=1)

        # 3) Combined weighted sum
        return self.weight_ms_ssim * ms_loss + self.weight_l1 * l1_loss

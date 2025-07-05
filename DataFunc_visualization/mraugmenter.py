# 파일명: mraugmenter.py (Scaling, Shearing 제거 버전)

import torch
import numpy as np
from math import exp
import torchvision.transforms.functional as TF

# --- 토치 버전 호환성을 위한 InterpolationMode 설정 ---
try:
    from torchvision.transforms.functional import InterpolationMode
    BILINEAR = InterpolationMode.BILINEAR
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BILINEAR = 'bilinear'
    BICUBIC = 'bicubic'


class MRAugmenter:
    """
    MRI k-space 데이터에 대해 물리적 특성을 고려한 데이터 증강을 수행하는 클래스.
    """

    def __init__(self, config):
        self.config = config
        self.rng = np.random.RandomState()

    def schedule_p(self, current_epoch):
        """현재 에포크에 따라 증강 확률(p)을 계산합니다."""
        if not self.config.aug_on:
            return 0.0

        t = current_epoch
        D = self.config.aug_delay
        T = self.config.max_epochs
        p_max = self.config.aug_strength

        if t < D:
            return 0.0

        schedule = self.config.aug_schedule
        if schedule == 'constant':
            p = p_max
        elif schedule == 'ramp':
            p = min(p_max, (t - D) / (T - D) * p_max) if T > D else p_max
        elif schedule == 'exp':
            if T <= D:
                return p_max
            c = self.config.aug_exp_decay
            p = p_max * (1 - exp(-c * (t - D) / (T - D))) / (1 - exp(-c))
        else:
            raise ValueError(f"알 수 없는 스케줄 방식입니다: {schedule}")

        return np.clip(p, 0.0, 1.0)

    def _random_apply(self, transform_name, p):
        """주어진 확률(p)과 가중치에 따라 변환 적용 여부를 결정합니다."""
        weight = self.config.weight_dict.get(transform_name, 0.0)
        return self.rng.uniform() < (weight * p)

    def _fft(self, image):
        """이미지 공간을 k-space로 변환합니다. (fftshift 적용)"""
        kspace_uncentered = torch.fft.fft2(image, norm='ortho')
        return torch.fft.fftshift(kspace_uncentered, dim=(-2, -1))

    def _rss(self, image):
        """다중 코일 이미지를 RSS(Root-Sum-of-Squares)로 결합합니다."""
        return torch.sqrt(torch.sum(torch.abs(image) ** 2, dim=0))
    
    def _apply_transforms(self, image_tensor, p):
        """하나의 복소수 이미지 텐서에 모든 증강 변환을 순차적으로 적용합니다."""
        img_real_view = torch.view_as_real(image_tensor).permute(0, 3, 1, 2).reshape(-1, image_tensor.shape[-2], image_tensor.shape[-1])
        
        # --- 1. 픽셀 보존 변환 ---
        if self._random_apply('fliph', p):
            img_real_view = TF.hflip(img_real_view)
        if self._random_apply('flipv', p):
            img_real_view = TF.vflip(img_real_view)
 

        

        
        C, H, W = image_tensor.shape
        aug_image_tensor = torch.view_as_complex(img_real_view.reshape(C, 2, H, W).permute(0, 2, 3, 1).contiguous())

        return aug_image_tensor

    def __call__(self, kspace_slice, target_size, current_epoch):
        """주어진 k-space 슬라이스에 증강 파이프라인을 적용합니다."""
        if isinstance(kspace_slice, np.ndarray):
            kspace_slice = torch.from_numpy(kspace_slice).cfloat()

        p = self.schedule_p(current_epoch)

        # 입력 k-space가 어떤 상태이든, 항상 올바른 중앙 이미지를 생성
        image_domain = torch.fft.ifft2(kspace_slice, norm='ortho')
        image = torch.fft.fftshift(image_domain, dim=(-2, -1))

        if p > 0:
            aug_image = self._apply_transforms(image, p)
        else:
            aug_image = image

        aug_kspace = self._fft(aug_image)
        aug_target = self._rss(aug_image)
        aug_target = TF.center_crop(aug_target.unsqueeze(0), output_size=target_size).squeeze(0)
        
        return aug_kspace, aug_target
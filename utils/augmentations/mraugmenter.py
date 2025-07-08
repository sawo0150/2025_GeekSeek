# 파일명: mraugmenter.py (메서드 분리 및 회전/확대 기능 추가 버전)

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
    각 증강 기법이 별도의 메서드로 분리되어 모듈성이 향상되었습니다.
    """
    def __init__(self, aug_on, aug_delay, max_epochs, aug_strength, aug_schedule,
                aug_exp_decay, weight_dict, max_rotation_angle, scale_range):
        
        """
        MRAugmenter를 초기화합니다.

        Args:
         (object): 증강 관련 하이퍼파라미터를 담고 있는 설정 객체.
                             필요한 속성:
                             - aug_on (bool): 증강 사용 여부
                             - aug_delay (int): 증강 시작 전 딜레이 에포크
                             - max_epochs (int): 총 학습 에포크
                             - aug_strength (float): 최대 증강 확률 (p_max)
                             - aug_schedule (str): 확률 스케줄 방식 ('constant', 'ramp', 'exp')
                             - aug_exp_decay (float): 'exp' 스케줄 사용 시 감쇠 계수
                             - weight_dict (dict): 각 증강의 가중치 딕셔너리
                                 (e.g., {'fliph': 1.0, 'flipv': 1.0, 'rotate': 0.5, 'scale': 0.5})
                             - max_rotation_angle (float): 최대 회전 각도 (도 단위)
                             - scale_range (tuple): 확대/축소 비율 범위 (e.g., (0.8, 1.2))
        """
        """ MRAugmenter를 초기화합니다. (Hydra에 최적화됨) """
        # 모든 파라미터를 self에 저장
        self.aug_on = aug_on
        self.aug_delay = aug_delay
        self.max_epochs = max_epochs
        self.aug_strength = aug_strength
        self.aug_schedule = aug_schedule
        self.aug_exp_decay = aug_exp_decay
        self.weight_dict = weight_dict
        self.max_rotation_angle = max_rotation_angle
        self.scale_range = scale_range
        
        self.rng = np.random.RandomState()

    def schedule_p(self, current_epoch):
        """현재 에포크에 따라 증강 확률(p)을 계산합니다."""
        if not self.aug_on:
            return 0.0

        t = current_epoch
        D = self.aug_delay
        T = self.max_epochs
        p_max = self.aug_strength

        if t < D:
            return 0.0

        schedule = self.aug_schedule
        if schedule == 'constant':
            p = p_max
        elif schedule == 'ramp':
            p = min(p_max, (t - D) / (T - D) * p_max) if T > D else p_max
        elif schedule == 'exp':
            if T <= D:
                return p_max
            c = self.aug_exp_decay
            p = p_max * (1 - exp(-c * (t - D) / (T - D))) / (1 - exp(-c))
        else:
            raise ValueError(f"알 수 없는 스케줄 방식입니다: {schedule}")

        return np.clip(p, 0.0, 1.0)

    def _random_apply(self, transform_name, p):
        """주어진 확률(p)과 가중치에 따라 변환 적용 여부를 결정합니다."""
        weight = self.weight_dict.get(transform_name, 0.0)
        return self.rng.uniform() < (weight * p)

    def _fft(self, image):
        """이미지 공간을 k-space로 변환합니다. (fftshift 적용)"""
        kspace_uncentered = torch.fft.fft2(image, norm='ortho')
        return torch.fft.fftshift(kspace_uncentered, dim=(-2, -1))

    def _rss(self, image):
        """다중 코일 이미지를 RSS(Root-Sum-of-Squares)로 결합합니다."""
        return torch.sqrt(torch.sum(torch.abs(image) ** 2, dim=0))

    # --- 개별 증강 메서드 ---

    def _transform_hflip(self, image_tensor):
        """수평 뒤집기. 복소수 텐서를 입력받아 처리합니다."""
        img_real_view = torch.view_as_real(image_tensor).permute(0, 3, 1, 2).reshape(-1, image_tensor.shape[-2], image_tensor.shape[-1])
        flipped_real_view = TF.hflip(img_real_view)
        C, H, W = image_tensor.shape
        return torch.view_as_complex(flipped_real_view.reshape(C, 2, H, W).permute(0, 2, 3, 1).contiguous())

    def _transform_vflip(self, image_tensor):
        """수직 뒤집기. 복소수 텐서를 입력받아 처리합니다."""
        img_real_view = torch.view_as_real(image_tensor).permute(0, 3, 1, 2).reshape(-1, image_tensor.shape[-2], image_tensor.shape[-1])
        flipped_real_view = TF.vflip(img_real_view)
        C, H, W = image_tensor.shape
        return torch.view_as_complex(flipped_real_view.reshape(C, 2, H, W).permute(0, 2, 3, 1).contiguous())

    def _transform_rotate(self, image_tensor):
        """회전. 복소수 텐서를 입력받아 처리하며, 물리적 타당성을 위해 보간을 포함합니다."""
        angle = self.rng.uniform(-self.max_rotation_angle, self.max_rotation_angle)
        
        img_real_view = torch.view_as_real(image_tensor).permute(0, 3, 1, 2).reshape(-1, image_tensor.shape[-2], image_tensor.shape[-1])
        rotated_real_view = TF.rotate(img_real_view, angle, interpolation=BILINEAR)
        C, H, W = image_tensor.shape
        return torch.view_as_complex(rotated_real_view.reshape(C, 2, H, W).permute(0, 2, 3, 1).contiguous())

    def _transform_scale(self, image_tensor):
        """확대/축소. 복소수 텐서를 입력받아 처리하며, 물리적 타당성을 위해 보간을 포함합니다."""
        C, H, W = image_tensor.shape
        scale_factor = self.rng.uniform(*self.scale_range)
        
        new_H, new_W = int(H * scale_factor), int(W * scale_factor)

        img_real_view = torch.view_as_real(image_tensor).permute(0, 3, 1, 2).reshape(-1, H, W)
        
        # 리사이즈 후 중앙 크롭으로 확대/축소 효과 구현
        resized_real_view = TF.resize(img_real_view, size=[new_H, new_W], interpolation=BILINEAR)
        cropped_real_view = TF.center_crop(resized_real_view, output_size=[H, W])
        
        return torch.view_as_complex(cropped_real_view.reshape(C, 2, H, W).permute(0, 2, 3, 1).contiguous())

    def _apply_transforms(self, image_tensor, p):
        """
        하나의 복소수 이미지 텐서에 모든 증강 변환을 순차적으로 적용합니다.
        각 변환은 모듈화된 별도의 메서드로 호출됩니다.
        """
        # --- 1. 픽셀 보존 변환 (보간 불필요) ---
        if self._random_apply('fliph', p):
            image_tensor = self._transform_hflip(image_tensor)
        if self._random_apply('flipv', p):
            image_tensor = self._transform_vflip(image_tensor)

        # --- 2. 픽셀 비보존 변환 (보간 필요) ---
        if self._random_apply('rotate', p):
            image_tensor = self._transform_rotate(image_tensor)
        if self._random_apply('scale', p):
            image_tensor = self._transform_scale(image_tensor)
        
        # 여기에 다른 증강 메서드 호출을 추가할 수 있습니다.
        
        return image_tensor

    # def __call__(self, kspace_slice, target_size, current_epoch):
    #     """주어진 k-space 슬라이스에 증강 파이프라인을 적용합니다."""
    #     if isinstance(kspace_slice, np.ndarray):
    #         kspace_slice = torch.from_numpy(kspace_slice).cfloat()

    #     p = self.schedule_p(current_epoch)

    #     # 입력 k-space로부터 이미지 공간으로 변환 (iFFT + fftshift)
    #     image_domain = torch.fft.ifft2(kspace_slice, norm='ortho')
    #     image = torch.fft.fftshift(image_domain, dim=(-2, -1))

    #     # 증강 확률에 따라 변환 적용
    #     if p > 0:
    #         aug_image = self._apply_transforms(image, p)
    #     else:
    #         aug_image = image

    #     # 다시 k-space로 변환하고 타겟 이미지 생성
    #     aug_kspace = self._fft(aug_image)
    #     aug_target = self._rss(aug_image)
    #     aug_target = TF.center_crop(aug_target.unsqueeze(0), output_size=target_size).squeeze(0)
        
    #     return aug_kspace, aug_target
    
    def __call__(self, kspace_slice, target_size, current_epoch):
        """주어진 k-space 슬라이스에 증강 파이프라인을 적용합니다."""
        if isinstance(kspace_slice, np.ndarray):
            kspace_slice = torch.from_numpy(kspace_slice).cfloat()

        p = self.schedule_p(current_epoch)

        # --- 올바른 이미지 공간 변환 ---
        # 1. k-space의 중앙에 있는 저주파 성분을 ifft2를 위해 꼭짓점으로 이동 (ifftshift)
        kspace_unshifted = torch.fft.ifftshift(kspace_slice, dim=(-2, -1))
        
        # 2. 역 푸리에 변환을 통해 이미지 공간으로 이동
        image_domain = torch.fft.ifft2(kspace_unshifted, norm='ortho')
        
        # 3. 4분면이 뒤섞인 이미지를 중앙으로 정렬 (fftshift)
        image = torch.fft.fftshift(image_domain, dim=(-2, -1))
        # --- 여기까지 수정 ---

        # 증강 확률에 따라 변환 적용
        if p > 0:
            aug_image = self._apply_transforms(image, p)
        else:
            aug_image = image

        # 다시 k-space로 변환하고 타겟 이미지 생성
        # (_fft 메서드는 내부적으로 fft -> fftshift 순서로 올바르게 구현되어 있음)
        aug_kspace = self._fft(aug_image)
        aug_target = self._rss(aug_image)
        aug_target = TF.center_crop(aug_target.unsqueeze(0), output_size=target_size).squeeze(0)
        
        return aug_kspace, aug_target
    


# class Config:
#     def __init__(self):
#         self.aug_on = True
#         self.max_epochs = 100
#         self.aug_delay = 10
#         self.aug_strength = 0.9
#         self.aug_schedule = 'ramp'
#         self.aug_exp_decay = 6.0
        
#         # 각 증강에 대한 가중치 설정 (새로운 증강 'rotate', 'scale' 추가)
#         self.weight_dict = {
#             'fliph': 0.0,
#             'flipv': 0.0,
#             'rotate': 1.0,
#             'scale': 1.0,
#         }
        
#         # 새로운 증강을 위한 하이퍼파라미터
#         self.max_rotation_angle = 15.0  # 최대 15도까지 회전
#         self.scale_range = (0.85, 1.15) # 85% ~ 115% 크기로 확대/축소
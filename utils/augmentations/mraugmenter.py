# 파일명: mraugmenter.py (val_loss 스케줄러 추가 및 데이터로더 통합 버전)

import torch
import numpy as np
from math import exp
from collections import deque
import torchvision.transforms.functional as TF

# --- 토치 버전 호환성을 위한 InterpolationMode 설정 ---
try:
    from torchvision.transforms.functional import InterpolationMode
    BILINEAR = InterpolationMode.BILINEAR
except ImportError:
    BILINEAR = 'bilinear'


class MRAugmenter:
    """
    MRI k-space 데이터에 대해 물리적 특성을 고려한 데이터 증강을 수행하는 클래스.
    epoch 진행률 또는 validation loss 기울기에 따라 증강 확률을 동적으로 조절합니다.
    데이터 로더의 변환(transform) 파이프라인에 통합되어 사용되도록 설계되었습니다.
    """
    def __init__(self, aug_on, aug_strength, aug_schedule_mode, aug_schedule_type,
                 aug_delay, max_epochs, aug_exp_decay,
                 val_loss_window_size, val_loss_grad_start, val_loss_grad_plateau,
                 weight_dict, max_rotation_angle, scale_range,
                 shift_extent, max_shear_angle):
        """
        MRAugmenter를 초기화합니다. (Hydra에 최적화됨)

        Args:
            aug_on (bool): 증강 사용 여부
            aug_strength (float): 최대 증강 확률 (p_max)
            aug_schedule_mode (str): 스케줄러 모드 ('constant', 'epoch', 'val_loss')
            aug_schedule_type (str): 확률 스케일링 방식 ('ramp', 'exp')
            aug_delay (int): 증강 시작 전 딜레이 에포크
            max_epochs (int): 총 학습 에포크 ('epoch' 모드에서 사용)
            aug_exp_decay (float): 'exp' 스케줄 사용 시 감쇠 계수
            val_loss_window_size (int): val_loss 기울기 계산 시 사용할 에포크 창 크기
            val_loss_grad_start (float): p=0으로 간주할 val_loss 기울기 시작점
            val_loss_grad_plateau (float): p=p_max로 간주할 val_loss 기울기 평탄화 지점
            weight_dict (dict): 각 증강의 가중치
            max_rotation_angle (float): 최대 회전 각도
            scale_range (tuple): 확대/축소 비율 범위
            shift_extent (float): 이동 증강의 표준편차 (픽셀 단위)
            max_shear_angle (float): 전단 변환의 최대 각도 (도 단위)
        """
        # 모든 파라미터를 self에 저장
        self.aug_on = aug_on
        self.aug_strength = aug_strength
        self.aug_schedule_mode = aug_schedule_mode
        self.aug_schedule_type = aug_schedule_type
        self.aug_delay = aug_delay
        self.max_epochs = max_epochs
        self.aug_exp_decay = aug_exp_decay
        self.val_loss_window_size = val_loss_window_size
        self.val_loss_grad_start = val_loss_grad_start
        self.val_loss_grad_plateau = val_loss_grad_plateau
        self.weight_dict = weight_dict
        self.max_rotation_angle = max_rotation_angle
        self.scale_range = scale_range
        self.shift_extent = shift_extent
        self.max_shear_angle = max_shear_angle

        self.rng = np.random.RandomState()
        
        # ✨ 상태 저장을 위한 변수 초기화
        self.current_epoch = 0
        self.val_loss_history = deque(maxlen=self.val_loss_window_size)

    def update_state(self, current_epoch, val_loss=None):
        """매 에폭 시작 시 호출되어 스케줄러의 현재 상태를 업데이트합니다."""
        self.current_epoch = current_epoch
        if val_loss is not None:
            self.val_loss_history.append(val_loss)

    def schedule_p(self):
        """현재 상태(epoch, val_loss)에 따라 증강 확률(p)을 계산합니다."""
        if not self.aug_on or self.current_epoch < self.aug_delay:
            return 0.0

        p_max = self.aug_strength
        schedule_mode = self.aug_schedule_mode

        if schedule_mode == 'constant':
            return p_max
        
        # 정규화된 진행률(k) 계산 (0.0 ~ 1.0)
        k = 0.0
        if schedule_mode == 'epoch':
            T, D = self.max_epochs, self.aug_delay
            k = (self.current_epoch - D) / (T - D) if T > D else 1.0
        
        elif schedule_mode == 'val_loss':
            if len(self.val_loss_history) < 2:
                return 0.0  # 기울기 계산에 최소 2개의 데이터 포인트 필요
            
            history = np.array(list(self.val_loss_history))
            epochs_indices = np.arange(len(history))
            
            # 선형 회귀를 통해 기울기(slope) 계산
            slope, _ = np.polyfit(epochs_indices, history, 1)

            # 기울기를 0~1 사이의 진행률 k로 정규화
            k = (slope - self.val_loss_grad_start) / (self.val_loss_grad_plateau - self.val_loss_grad_start)
        else:
            raise ValueError(f"알 수 없는 스케줄 모드입니다: {schedule_mode}")
        
        k = np.clip(k, 0.0, 1.0)

        # 진행률 k를 기반으로 최종 확률 p 계산
        schedule_type = self.aug_schedule_type
        if schedule_type == 'ramp':
            p = p_max * k
        elif schedule_type == 'exp':
            c = self.aug_exp_decay
            # k=0일 때 p=0, k=1일 때 p=p_max가 되도록 스케일링
            p = p_max * (1 - exp(-c * k)) / (1 - exp(-c)) if c > 0 else p_max * k
        else:
            raise ValueError(f"알 수 없는 스케줄 타입입니다: {schedule_type}")

        return np.clip(p, 0.0, 1.0)

    # ... (내부 헬퍼 및 개별 증강 메서드는 이전과 동일) ...
    def _random_apply(self, transform_name, p):
        weight = self.weight_dict.get(transform_name, 0.0)
        return self.rng.uniform() < (weight * p)

    def _fft(self, image):
        kspace_uncentered = torch.fft.fft2(image, norm='ortho')
        return torch.fft.fftshift(kspace_uncentered, dim=(-2, -1))

    def _rss(self, image):
        return torch.sqrt(torch.sum(torch.abs(image) ** 2, dim=0))

    def _transform_hflip(self, image_tensor):
        img_real_view = torch.view_as_real(image_tensor).permute(0, 3, 1, 2).reshape(-1, image_tensor.shape[-2], image_tensor.shape[-1])
        flipped_real_view = TF.hflip(img_real_view)
        C, H, W = image_tensor.shape
        return torch.view_as_complex(flipped_real_view.reshape(C, 2, H, W).permute(0, 2, 3, 1).contiguous())

    def _transform_vflip(self, image_tensor):
        img_real_view = torch.view_as_real(image_tensor).permute(0, 3, 1, 2).reshape(-1, image_tensor.shape[-2], image_tensor.shape[-1])
        flipped_real_view = TF.vflip(img_real_view)
        C, H, W = image_tensor.shape
        return torch.view_as_complex(flipped_real_view.reshape(C, 2, H, W).permute(0, 2, 3, 1).contiguous())

    def _transform_rotate(self, image_tensor):
        angle = self.rng.uniform(-self.max_rotation_angle, self.max_rotation_angle)
        img_real_view = torch.view_as_real(image_tensor).permute(0, 3, 1, 2).reshape(-1, image_tensor.shape[-2], image_tensor.shape[-1])
        rotated_real_view = TF.rotate(img_real_view, angle, interpolation=BILINEAR)
        C, H, W = image_tensor.shape
        return torch.view_as_complex(rotated_real_view.reshape(C, 2, H, W).permute(0, 2, 3, 1).contiguous())

    def _transform_scale(self, image_tensor):
        C, H, W = image_tensor.shape
        scale_factor = self.rng.uniform(*self.scale_range)
        new_H, new_W = int(H * scale_factor), int(W * scale_factor)
        img_real_view = torch.view_as_real(image_tensor).permute(0, 3, 1, 2).reshape(-1, H, W)
        resized_real_view = TF.resize(img_real_view, size=[new_H, new_W], interpolation=BILINEAR)
        cropped_real_view = TF.center_crop(resized_real_view, output_size=[H, W])
        return torch.view_as_complex(cropped_real_view.reshape(C, 2, H, W).permute(0, 2, 3, 1).contiguous())

    def _transform_shift(self, image_tensor):
        """정규분포에 따라 이미지 상하/좌우 순환 이동. 보간 없음."""
        # self.shift_extent를 표준편차로 사용하여 이동 거리 샘플링
        shift_y = self.rng.normal(0, self.shift_extent)
        shift_x = self.rng.normal(0, self.shift_extent)
        
        # torch.roll은 정수형 입력을 받으므로 반올림
        shifts = (int(round(shift_y)), int(round(shift_x)))
        
        # dims=(-2, -1)은 각각 H, W 차원을 의미
        return torch.roll(image_tensor, shifts=shifts, dims=(-2, -1))

    # ┕ [추가] 전단(Shear) 변환 메서드
    def _transform_shear(self, image_tensor):
        """이미지를 평행사변형으로 변환. 보간 사용."""
        # -max_shear_angle ~ +max_shear_angle 범위에서 전단 각도 샘플링
        shear_angle = self.rng.uniform(-self.max_shear_angle, self.max_shear_angle)
        
        # affine 변환을 위해 real/imag 채널로 분리
        img_real_view = torch.view_as_real(image_tensor).permute(0, 3, 1, 2).reshape(-1, image_tensor.shape[-2], image_tensor.shape[-1])
        
        # affine 변환 적용 (shear 인자 사용)
        sheared_real_view = TF.affine(img_real_view, angle=0, translate=[0, 0], scale=1.0, shear=[shear_angle], interpolation=BILINEAR)
        
        # 다시 복소수 텐서로 변환
        C, H, W = image_tensor.shape
        return torch.view_as_complex(sheared_real_view.reshape(C, 2, H, W).permute(0, 2, 3, 1).contiguous())


    def _apply_transforms(self, image_tensor, p):
        # [수정] 어떤 증강이 적용되었는지 로그를 남김
        applied_transforms = []
        if self._random_apply('fliph', p):
            image_tensor = self._transform_hflip(image_tensor)
            applied_transforms.append('hflip')
        if self._random_apply('flipv', p):
            image_tensor = self._transform_vflip(image_tensor)
            applied_transforms.append('flipv')
        if self._random_apply('rotate', p):
            image_tensor = self._transform_rotate(image_tensor)
            applied_transforms.append('rotate')
        if self._random_apply('scale', p):
            image_tensor = self._transform_scale(image_tensor)
            applied_transforms.append('scale')
        if self._random_apply('shift', p):
            image_tensor = self._transform_shift(image_tensor)
            applied_transforms.append('shift')
        if self._random_apply('shear', p):
            image_tensor = self._transform_shear(image_tensor)
            applied_transforms.append('shear')
            
        if applied_transforms:
            print(f"[Augmenter] Applied: {', '.join(applied_transforms)}")
            
        return image_tensor

    def __call__(self, mask, kspace_np, target_np, attrs, fname, slice_idx):
        """
        데이터 로더의 transform 파이프라인의 일부로 동작합니다.
        numpy 배열을 입력받아 증강을 적용하고, 다시 numpy 배열을 반환합니다.
        """
        p = self.schedule_p()
        if p == 0:
            # 증강이 비활성화된 경우, 원본 데이터를 그대로 반환
            return mask, kspace_np, target_np, attrs, fname, slice_idx

        kspace_slice = torch.from_numpy(kspace_np).cfloat()

        # k-space -> image 공간으로 변환
        kspace_unshifted = torch.fft.ifftshift(kspace_slice, dim=(-2, -1))
        image_domain = torch.fft.ifft2(kspace_unshifted, norm='ortho')
        image = torch.fft.fftshift(image_domain, dim=(-2, -1))

        # 증강 적용
        aug_image = self._apply_transforms(image, p)

        # 증강된 이미지로부터 새로운 k-space와 target 생성
        aug_kspace = self._fft(aug_image)
        aug_target = self._rss(aug_image)

        # 다음 transform(e.g., cropping)을 위해 numpy 배열로 변환하여 반환
        return mask, aug_kspace.numpy(), aug_target.numpy(), attrs, fname, slice_idx
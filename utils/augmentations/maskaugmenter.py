import numpy as np, random
from typing import Dict, Any, Sequence
from utils.data.subsample import create_mask_for_mask_type
import torch

class MaskAugmenter:
    """
    * MRaugmenter 와 동일한 스케줄러 로직을 사용해 확률 p 결정.
    * p 가 ‘당첨’되면:
        ① mask type 을 가중치(prob) 로 선택
        ② accel / cf 를 range·list 규칙에 따라 샘플
        ③ subsample.create_mask_for_mask_type 로 MaskFunc 생성
        ④ 새 mask 를 만들어 반환 (k-space 는 그대로 → MaskApplyTransform 이 곱함)
    """
    def __init__(self,
                 aug_on: bool,
                 aug_strength: float,
                 aug_schedule_mode: str, aug_schedule_type: str,
                 aug_delay: int, max_epochs: int, aug_exp_decay: float,
                 val_loss_window_size: int, val_loss_grad_start: float,
                 val_loss_grad_plateau: float,
                 mask_specs: Dict[str, Dict[str, Any]],
                 allow_any_combination: bool = True):
        # --- 스케줄 파라미터 저장 -----------------------------------------
        self.aug_on = aug_on
        # print("ㄴㅇㄹㄴㅇㅎㄱㅎㅈㄷㄷㅈㅎㅈㄷ",aug_on)
        self.aug_strength = aug_strength
        self.mode  = aug_schedule_mode
        self.stype = aug_schedule_type
        self.delay = aug_delay
        self.T     = max_epochs
        self.decay = aug_exp_decay
        self.win   = val_loss_window_size
        self.grad0 = val_loss_grad_start
        self.grad1 = val_loss_grad_plateau
        # --- Mask spec -----------------------------------------
        self.mask_specs = mask_specs
        self.allow_any  = allow_any_combination
        # 확률 벡터 계산
        probs = [spec.get("prob", 1.0) for spec in mask_specs.values()]
        s = sum(probs)
        self.type_probs = [p/s for p in probs]
        self.types = list(mask_specs.keys())
        # 상태
        self.rng = np.random.RandomState()
        self.current_epoch = 0
        self.val_hist = []

    # -------------------------- 상태 업데이트 -----------------------------
    def update_state(self, current_epoch: int, val_loss=None):
        self.current_epoch = current_epoch
        if val_loss is not None:
            self.val_hist.append(val_loss)
            if len(self.val_hist) > self.win:
                self.val_hist.pop(0)

    # -------------------------- 확률 스케줄 -------------------------------
    def _prob_now(self):
        if (not self.aug_on) or (self.current_epoch < self.delay):
            return 0.0
        p_max = self.aug_strength
        if self.mode == "constant":
            return p_max
        # k ∈ [0,1]
        if self.mode == "epoch":
            k = (self.current_epoch - self.delay) / max(1, self.T - self.delay)
        elif self.mode == "val_loss" and len(self.val_hist) >= 2:
            x = np.arange(len(self.val_hist))
            slope, _ = np.polyfit(x, np.array(self.val_hist), 1)
            k = (slope - self.grad0) / (self.grad1 - self.grad0)
        else:
            k = 0.0
        k = np.clip(k, 0.0, 1.0)
        if self.stype == "ramp":
            return p_max * k
        elif self.stype == "exp":
            return p_max * (1 - np.exp(-self.decay * k)) / (1 - np.exp(-self.decay))
        else:
            raise ValueError(f"unknown schedule type {self.stype}")

    # -------------------------- 샘플러 유틸 -------------------------------
    def _sample_param(self, spec):
        # 연속 범위 [a,b]
        if isinstance(spec, Sequence) and len(spec) == 2 and all(isinstance(v,(int,float)) for v in spec):
            a,b = spec
            if all(isinstance(v,int) for v in spec):
                return self.rng.randint(a, b+1)
            return self.rng.uniform(a, b)
        # 리스트 [v1,v2,...]
        if isinstance(spec, Sequence):
            return self.rng.choice(spec)
        # 단일 값
        return spec

    def _make_mask(self, shape) -> np.ndarray:
        # ① mask type 선택
        mtype = self.rng.choice(self.types, p=self.type_probs)
        spec  = self.mask_specs[mtype]
        # ② 파라미터 샘플링
        accel = self._sample_param(spec["accel"])
        cf    = self._sample_param(spec["cf"])
        mf = create_mask_for_mask_type(
            mtype,
            center_fractions=[cf],
            accelerations=[accel],
        )
        if hasattr(mf, "allow_any_combination"):
            mf.allow_any_combination = self.allow_any
        mask, _ = mf(shape)          # torch.Tensor
        return mask.numpy()

    # -------------------------- core -------------------------------------
    def __call__(self, mask, kspace_np, target_np, attrs, fname, slice_idx, *extra):
        if self.rng.rand() >= self._prob_now():
            return mask, kspace_np, target_np, attrs, fname, slice_idx, *extra
        # 새 mask 생성 (shape = kspace[..., H, W])
        new_mask = self._make_mask(kspace_np.shape)
        return new_mask, kspace_np, target_np, attrs, fname, slice_idx, *extra

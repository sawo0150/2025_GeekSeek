# ----------------------------------------------------------------------
#   DuplicateMaskDataset  –  SliceData 를 “가속도별로 복제”하는 래퍼
# ----------------------------------------------------------------------
from torch.utils.data import Dataset
import torch          # new_mask 타입 체크용 (Optional)
from fastmri.data.subsample import create_mask_for_mask_type
import numpy as np
from collections.abc import Sequence       # 추가

class DuplicateMaskDataset(Dataset):
    """
    base_ds[idx] 가 반환하는 7-tuple
      (mask, kspace, target, maximum, fname, slice_idx, cat)
    을 *accel_cfgs* 길이만큼 복제한다.
    각 복제본은
      • 새로 샘플한 mask (accel, cf 범위, mask_type 기준)
      • cat 문자열의 ‘_x4’/‘_x8’ 부분을 가속도에 맞춰 교체
    로 교체되어 리턴된다.
    """
    def __init__(self,
                 base_ds: Dataset,
                 accel_cfgs,
                 allow_any_combination: bool = True):
        """
        accel_cfgs: List[Dict]
            └─ {accel: 4, cf: [0.07,0.12], mask_type: random}
               {accel: 8, cf: 0.04,       mask_type: equispaced}
        """
        self.base_ds   = base_ds
        self.cfgs      = accel_cfgs
        self.dup       = len(accel_cfgs)
        self.allow_any = allow_any_combination
        self.rng       = np.random.RandomState()

        # ▶ coil_counts 도 길이를 dup배로 맞춰준다
        if hasattr(base_ds, "coil_counts"):
            self.coil_counts = np.repeat(base_ds.coil_counts, self.dup).tolist()

    # --------------------------------------------------------------
    def __len__(self):
        return len(self.base_ds) * self.dup

    # ---------------- 헬퍼: 파라미터 샘플링 ------------------------
    def _sample_param(self, spec):
        # if isinstance(spec, (list, tuple)) and len(spec) == 2:      # [min,max] 연속
        #     a, b = spec
        #     if all(isinstance(v, int) for v in spec):
        #         return self.rng.randint(a, b + 1)
        #     return self.rng.uniform(a, b)
        # if isinstance(spec, (list, tuple)):                         # choice
        #     return self.rng.choice(spec)

       # OmegaConf ListConfig 도 잡도록 Sequence 로 체크
        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
            spec_list = list(spec)      # → 파이썬 list 로 변환
            # --- [min,max] 형태 (연속 범위) ---------------------------
            if len(spec_list) == 2 and all(isinstance(v, (int, float)) for v in spec_list):
                a, b = spec_list
                if all(isinstance(v, int) for v in spec_list):
                    return self.rng.randint(a, b + 1)
                return self.rng.uniform(a, b)
            # --- 선택 리스트 [v1, v2, …] ------------------------------
            return self.rng.choice(spec_list)

        return spec                                                 # 단일 값

    # --------------------------------------------------------------
    def __getitem__(self, idx):
        base_idx  = idx // self.dup
        cfg_idx   = idx %  self.dup
        cfg       = self.cfgs[cfg_idx]

        # ① 원본 샘플 가져오기
        mask, kspace, target, maximum, fname, slice_idx, cat = self.base_ds[base_idx]
        # print(mask)
        # ② 새 mask 생성
        accel = cfg["accel"]
        cf    = self._sample_param(cfg["cf"])
        mtype = cfg.get("mask_type", "random")

        mf = create_mask_for_mask_type(mtype, [cf], [accel])
        if hasattr(mf, "allow_any_combination"):
            mf.allow_any_combination = self.allow_any
            

        # ── 버전마다 반환 형태가 다르므로 안전하게 처리 ───────────────
        # mask_res = mf(kspace.shape)
        # new_mask = mask_res[0] if isinstance(mask_res, tuple) else mask_res
        # if hasattr(new_mask, "numpy"):      # torch.Tensor → np.ndarray
        #     new_mask = new_mask.numpy()

        # MaskFunc 는 torch.Tensor(float32) 또는 tuple 반환 → 안전하게 numpy-bool 로 변환
        mask_res   = mf(kspace.shape)                         # (mask, n_lowfreq) or mask
        mask_tensor = mask_res[0] if isinstance(mask_res, tuple) else mask_res

        if isinstance(mask_tensor, torch.Tensor):
            mask_np = mask_tensor.cpu().numpy()
        else:
            mask_np = mask_tensor

        # 0/1 → uint8  (VarNet 은 이후 .bool()을 호출한 뒤 argmin 을 씁니다)
        new_mask = (mask_np > 0).astype(np.uint8)             # final dtype = uint8 

        # ③ cat 문자열 수정 (brain_x4 → brain_x8 등)
        try:
            organ, _ = cat.split("_")
            cat = f"{organ}_x{accel}"
        except ValueError:
            pass  # 예외 시 그대로

        return new_mask, kspace, target, maximum, fname, slice_idx, cat

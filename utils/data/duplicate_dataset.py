# ----------------------------------------------------------------------
# DuplicateMaskDataset – SliceData 를 “가속도별로 복제”하지만
#                      마스크는 번들(npz)에서 직접 로드
# ----------------------------------------------------------------------
from torch.utils.data import Dataset
import numpy as np, torch, os, json

class DuplicateMaskDataset(Dataset):
    """
    base_ds[idx] → (mask, kspace, target, maximum, fname, slice_idx, cat)

    accel_cfgs :  List[Dict]  예) [{accel:4}, {accel:8}]
        • dict 안에 bundle_path 키를 주면 그 경로의 npz 를 사용
        • 하나라도 누락되면 KeyError 로 바로 알려줌
    """
    def __init__(self,
                 base_ds: Dataset,
                 accel_cfgs,
                 bundle_path: str = "metaData/precomputed_masks.npz"):
        self.base_ds = base_ds
        self.cfgs    = accel_cfgs
        self.dup     = len(accel_cfgs)

        # ---------- 번들 로드 ----------
        npz_path = accel_cfgs[0].get("bundle_path", bundle_path)
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"mask bundle not found: {npz_path}")
        # self.bundle = np.load(npz_path, allow_pickle=True)
        with np.load(npz_path, allow_pickle=True) as z:
            self.bundle = {k: z[k] for k in z.files}

        # coil_counts 도 dup 배수로
        if hasattr(base_ds, "coil_counts"):
            self.coil_counts = np.repeat(base_ds.coil_counts,
                                         self.dup).tolist()

    # --------------------------------------------------------------
    def __len__(self):
        return len(self.base_ds) * self.dup

    # --------------------------------------------------------------
    def __getitem__(self, idx):
        base_idx  = idx // self.dup
        cfg_idx   = idx %  self.dup
        accel     = self.cfgs[cfg_idx]["accel"]

        # --- 원본 샘플 ---
        mask, kspace, target, maximum, fname, slice_idx, cat = (
            self.base_ds[base_idx])
        # print(mask.shape)
        # --- lookup key 만들기  (body_x{accel}_{N}) ---
        organ, _ = cat.split("_")           # brain_x4 → brain
        N = len(mask)        # 원본 readout 폭 (crop 前)
        key      = f"{organ}_x{accel}_{N}"

        if key not in self.bundle:
            print(cat, fname, kspace.shape)
            raise KeyError(f"mask '{key}' not in bundle npz")

        new_mask = self.bundle[key].astype(np.uint8)

        # --- cat 문자열 acc 수정 ---
        cat = f"{organ}_x{accel}"

        return new_mask, kspace, target, maximum, fname, slice_idx, cat

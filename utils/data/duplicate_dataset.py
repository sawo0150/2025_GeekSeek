# utils/data/duplicate_dataset.py
from torch.utils.data import Dataset
import numpy as np, torch, os, json

class DuplicateMaskDataset(Dataset):
    def __init__(self,
                 base_ds: Dataset,
                 accel_cfgs,
                 bundle_path: str = "metaData/precomputed_masks.npz"):
        self.base_ds = base_ds
        self.cfgs    = accel_cfgs
        self.dup     = len(accel_cfgs)

        npz_path = accel_cfgs[0].get("bundle_path", bundle_path)
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"mask bundle not found: {npz_path}")
        with np.load(npz_path, allow_pickle=True) as z:
            self.bundle = {k: z[k] for k in z.files}

        if hasattr(base_ds, "coil_counts"):
            self.coil_counts = np.repeat(base_ds.coil_counts, self.dup).tolist()
        if hasattr(base_ds, "sample_shapes"):
            self.sample_shapes = [shape for shape in base_ds.sample_shapes for _ in range(self.dup)]

    def __len__(self):
        return len(self.base_ds) * self.dup

    def __getitem__(self, idx):
        base_idx  = idx // self.dup
        cfg_idx   = idx %  self.dup
        accel     = self.cfgs[cfg_idx]["accel"]

        # [FIX] 8개 항목을 받도록 언패킹 수정
        mask, kspace, target, attrs, fname, slice_idx, cat, domain_idx = (
            self.base_ds[base_idx])
        
        organ, _ = cat.split("_")
        N = mask.shape[-1] # 원본 readout 폭
        key = f"{organ}_x{accel}_{N}"

        if key not in self.bundle:
            raise KeyError(f"mask '{key}' not in bundle npz")

        new_mask = self.bundle[key].astype(np.uint8)
        new_cat = f"{organ}_x{accel}"

        # [FIX] 8개 항목을 다시 반환 (domain_idx는 그대로 전달)
        return new_mask, kspace, target, attrs, fname, slice_idx, new_cat, domain_idx
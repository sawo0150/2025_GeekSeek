# utils/data/domain_subset.py
"""
SliceData 래퍼.
`allowed_cats`(예: ["knee_x4","knee_x8"]) 에 포함된 cat 만 노출한다.
"""
from torch.utils.data import Dataset

class DomainSubset(Dataset):
    def __init__(self, base_ds, allowed_cats):
        self.base_ds = base_ds
        self.allowed_cats = set(allowed_cats)
        # SliceData.kspace_examples = [(fname, slice_idx, cat), ...]
        self.idxs = [i for i, tpl in enumerate(base_ds.kspace_examples)
                     if tpl[2] in self.allowed_cats]
        # ★ sampler 에 필요한 속성 복제
        # 1) slice별 shape 리스트
        if hasattr(base_ds, "sample_shapes"):
            self.sample_shapes = [base_ds.sample_shapes[i] for i in self.idxs]
        # 2) coil 수 리스트
        if hasattr(base_ds, "coil_counts"):
            self.coil_counts   = [base_ds.coil_counts[i]   for i in self.idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        return self.base_ds[self.idxs[i]]

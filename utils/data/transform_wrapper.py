# utils/data/transform_wrapper.py
from torch.utils.data import Dataset

class TransformWrapper(Dataset):
    def __init__(self, base_ds: Dataset, transform):
        self.base_ds   = base_ds
        self.transform = transform
        if hasattr(base_ds, "coil_counts"): self.coil_counts = base_ds.coil_counts
        if hasattr(base_ds, "sample_shapes"): self.sample_shapes = base_ds.sample_shapes

    def __len__(self): return len(self.base_ds)

    def __getitem__(self, idx):
        sample = self.base_ds[idx]
        
        # [FIX] promptmr 모드(8-tuple)와 일반 모드(7-tuple)를 모두 처리
        if len(sample) == 8: # (mask, kspace, target, attrs, fname, sidx, cat, domain_idx)
            mask, kspace, target, attrs, fname, sidx, cat, domain_idx = sample
            # transform은 앞 6개 항목만 처리
            t_mask, t_kspace, t_target, t_maximum, t_fname, t_sidx = \
                self.transform(mask, kspace, target, attrs, fname, sidx)
            # transform 결과와 cat, domain_idx를 합쳐 반환
            return t_mask, t_kspace, t_target, t_maximum, t_fname, t_sidx, cat, domain_idx
        
        elif len(sample) == 7: # (mask, kspace, target, attrs, fname, sidx, cat)
            mask, kspace, target, attrs, fname, sidx, cat = sample
            t_mask, t_kspace, t_target, t_maximum, t_fname, t_sidx = \
                self.transform(mask, kspace, target, attrs, fname, sidx)
            return t_mask, t_kspace, t_target, t_maximum, t_fname, t_sidx, cat
            
        else: # isforward=True인 경우 등
            mask, kspace, target, attrs, fname, sidx = sample
            t_mask, t_kspace, t_target, t_maximum, t_fname, t_sidx = \
                self.transform(mask, kspace, target, attrs, fname, sidx)
            return t_mask, t_kspace, t_target, t_maximum, t_fname, t_sidx
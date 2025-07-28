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
        
        # [최종 수정] 9, 8, 7, 6개 모든 케이스를 처리하여 완벽한 호환성을 보장합니다.
        if len(sample) == 9: # promptmr + acc_idx 모드
            mask, kspace, target, attrs, fname, sidx, cat, domain_idx, acc_idx = sample
            t_mask, t_kspace, t_target, t_maximum, t_fname, t_sidx = \
                self.transform(mask, kspace, target, attrs, fname, sidx)
            return t_mask, t_kspace, t_target, t_maximum, t_fname, t_sidx, cat, domain_idx, acc_idx
        
        elif len(sample) == 8: # acc_idx가 없는 이전 promptmr 모드
            mask, kspace, target, attrs, fname, sidx, cat, domain_idx = sample
            t_mask, t_kspace, t_target, t_maximum, t_fname, t_sidx = \
                self.transform(mask, kspace, target, attrs, fname, sidx)
            return t_mask, t_kspace, t_target, t_maximum, t_fname, t_sidx, cat, domain_idx
        
        elif len(sample) == 7: # 일반 main.py 모드
            mask, kspace, target, attrs, fname, sidx, cat = sample
            t_mask, t_kspace, t_target, t_maximum, t_fname, t_sidx = \
                self.transform(mask, kspace, target, attrs, fname, sidx)
            return t_mask, t_kspace, t_target, t_maximum, t_fname, t_sidx, cat
            
        elif len(sample) == 6: # isforward=True 모드
            mask, kspace, target, attrs, fname, sidx = sample
            t_mask, t_kspace, t_target, t_maximum, t_fname, t_sidx = \
                self.transform(mask, kspace, target, attrs, fname, sidx)
            return t_mask, t_kspace, t_target, t_maximum, t_fname, t_sidx
        
        else:
            raise ValueError(f"Unexpected sample length from base_ds: {len(sample)}")
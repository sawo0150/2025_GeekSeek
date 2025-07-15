# utils/data/transform_wrapper.py
from torch.utils.data import Dataset

class TransformWrapper(Dataset):
    """
    base_ds 샘플(+cat) → transform_chain → 최종 (.., cat) 반환
    coil_counts 도 그대로 전달.
    """
    def __init__(self, base_ds: Dataset, transform):
        self.base_ds   = base_ds
        self.transform = transform

        # coil_counts 전달 (Sampler 호환)
        if hasattr(base_ds, "coil_counts"):
            self.coil_counts = base_ds.coil_counts
        # ✨ sample_shapes 전달 (GroupByCoilBatchSampler 호환)
        if hasattr(base_ds, "sample_shapes"):
            self.sample_shapes = base_ds.sample_shapes

    def __len__(self): return len(self.base_ds)

    def __getitem__(self, idx):
        # # base_ds 는 (mask,kspace,target,attrs,fname,sidx,cat)
        # mask,kc,tgt,attrs,fname,sidx,cat = self.base_ds[idx]

        # # transform_chain 은 cat 제외한 6-tuple 처리
        # mask,kc,tgt,maximum,fname,sidx = \
        #     self.transform(mask,kc,tgt,attrs,fname,sidx)

        # return mask,kc,tgt,maximum,fname,sidx,cat

        sample = self.base_ds[idx]
        if len(sample) == 7:                          # ─ train ─
            mask,kc,tgt,attrs,fname,sidx,cat = sample
            mask,kc,tgt,maximum,fname,sidx = \
                self.transform(mask,kc,tgt,attrs,fname,sidx)
            return mask,kc,tgt,maximum,fname,sidx,cat

        elif len(sample) == 6:                        # ─ forward ─
            mask,kc,tgt,attrs,fname,sidx   = sample
            mask,kc,tgt,maximum,fname,sidx = \
                self.transform(mask,kc,tgt,attrs,fname,sidx)
            return mask,kc,tgt,maximum,fname,sidx     # no cat

        else:
            raise ValueError(f"Unexpected sample length {len(sample)}")

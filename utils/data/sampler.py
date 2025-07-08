# utils/data/sampler.py
import random
from collections import defaultdict
from torch.utils.data import Sampler
from torch.utils.data.sampler import BatchSampler

class IdentitySampler(Sampler):
    """
    그냥 0..N-1 인덱스를 순서대로, shuffle=True 면 섞어서 리턴.
    """
    def __init__(self, data_source, shuffle: bool = False, **kwargs):
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        idxs = list(range(len(self.data_source)))
        if self.shuffle:
            random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return len(self.data_source)


class GroupByCoilBatchSampler(BatchSampler):
    """
    coil_counts 리스트를 보고,
    같은 coil 개수끼리 idx 를 batch_size 단위로 묶어서 리턴.
    """
    def __init__(self,
                 coil_counts,        # list of int (len == dataset size)
                 batch_size: int,
                 shuffle: bool = True,
                 **kwargs):
        # (BatchSampler 은 내부적으로 sampler, batch_size, drop_last 를 기대하지만
        #  우리는 override 하므로 super() 호출 안 해도 무방합니다.)
        self.batch_size = batch_size
        self.shuffle = shuffle

        # 1) coil별로 인덱스 그룹화
        groups = defaultdict(list)
        for idx, c in enumerate(coil_counts):
            groups[c].append(idx)

        # 2) 그룹 내 shuffle
        if shuffle:
            for g in groups.values():
                random.shuffle(g)

        # 3) batch_size 단위로 slice
        self.batches = []
        for g in groups.values():
            for i in range(0, len(g), batch_size):
                self.batches.append(g[i : i + batch_size])

        # 4) batches 순서 섞기
        if shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)

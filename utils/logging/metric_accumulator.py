# utils/logging/metric_accumulator.py
import re
from collections import defaultdict
try:
    import wandb
except ModuleNotFoundError:
    wandb = None

__all__ = ["MetricAccumulator"]

_CATS = ["knee_x4", "knee_x8", "brain_x4", "brain_x8"]

def _cat_from_fname(fname: str) -> str:
    organ = "knee" if "knee" in fname.lower() else "brain"
    acc   = "x4"   if re.search(r"acc4|x4|r04", fname, re.I) else "x8"
    return f"{organ}_{acc}"

class MetricAccumulator:
    """
    split = 'train' or 'val'
    update(loss, ssim, fnames)  # 배치마다 호출
    log(step)                   # epoch 막판에 한 번 호출
    """
    def __init__(self, split: str):
        self.split = split
        self.data = {c: defaultdict(float) for c in _CATS}
        self.total = defaultdict(float)

    def update(self, loss: float, ssim: float, fnames):
        # fnames: list[str] (배치 단위, 하나만 넘어와도 OK)
        for f in fnames:
            cat = _cat_from_fname(f)
            self.data[cat]["loss"] += loss
            self.data[cat]["ssim"] += ssim
            self.data[cat]["n"]   += 1
            self.total["loss"] += loss
            self.total["ssim"] += ssim
            self.total["n"] += 1

    def _avg(self, d):
        return d["loss"]/d["n"], d["ssim"]/d["n"]

    def log(self, step: int):
        if not wandb.run:  # W&B off
            return
        logdict = {}
        for cat, d in self.data.items():
            if d["n"] == 0:   # 해당 카테고리 샘플 X
                continue
            l, s = self._avg(d)
            logdict.setdefault(cat, {})[f"{self.split}_loss"] = l
            logdict[cat][f"{self.split}_ssim"] = s
        # overall
        l, s = self._avg(self.total)
        logdict.setdefault("overall", {})[f"{self.split}_loss"] = l
        logdict["overall"][f"{self.split}_ssim"] = s
        wandb.log(logdict, step=step)

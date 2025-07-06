# utils/logging/vis_logger.py
import re, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from collections import defaultdict     # ★ 이 줄 추가
try:
    import wandb          # 가급적 train 스크립트와 동일한 import 패턴
    
except ModuleNotFoundError:
    wandb = None

__all__ = ["log_epoch_samples", "make_figure"]

# ---------------------------------------------------------------------
# 공통 카테고리 추출 함수  (metric_accumulator 와 동일)
# ---------------------------------------------------------------------
def _cat_from_fname(fname: str) -> str:
    organ = "knee" if "knee" in fname.lower() else "brain"
    acc   = "x4"   if re.search(r"acc4|x4|r04", fname, re.I) else "x8"
    return f"{organ}_{acc}"

# ---------------------------------------------------------------------
# 6-pane Figure 생성 (Mag/QSM × GT/Rec/Err)
# ---------------------------------------------------------------------
def make_figure(
    m_gt, m_rec,
    q_gt=None, q_rec=None,
    title="",
    mag_clip_pct=0.995,
    qsm_max=None,            # None → π
    err_max=None,            # None → 자동(상위1% 값)
    overlay_alpha=0.6
):
    # ─── 준비 ──────────────────────────────────────────────────────
    m_err = np.abs(m_gt - m_rec)
    if q_gt is not None:
        q_err = np.abs(q_gt - q_rec)

    # ① Magnitude 범위 고정 (분위수 클리핑)
    def _mag_norm(im):
        lo, hi = np.quantile(im, [(1-mag_clip_pct)/2, 1-(1-mag_clip_pct)/2])
        return lo, hi if hi > lo else (im.min(), im.max())

    # ② Error max 고정
    if err_max is None:
        err_max = np.quantile(m_err, 0.99)   # 상위1%로 자동
    err_cmap = plt.cm.get_cmap("hot")        # 검→빨→노

    # ③ QSM 범위 고정
    if q_gt is not None:
        if qsm_max is None:
            qsm_max = np.pi
        qsm_cmap = plt.cm.get_cmap("seismic")

    # ─── Figure ───────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))

    # Magnitude GT / Rec
    for a, im, ttl in zip(ax[0, :2], [m_gt, m_rec], ["GT | Mag", "Rec | Mag"]):
        vmin, vmax = _mag_norm(im)
        a.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
        a.set_title(ttl, fontsize=10); a.axis("off")

    # Magnitude Error (overlay)
    a_err = ax[0, 2]
    vmin, vmax = _mag_norm(m_rec)
    a_err.imshow(m_rec, cmap="gray", vmin=vmin, vmax=vmax)
    a_err.imshow(m_err, cmap=err_cmap, vmin=0, vmax=err_max, alpha=overlay_alpha)
    a_err.set_title("Err | Mag", fontsize=10); a_err.axis("off")

    # === QSM 라인 ==================================================
    if q_gt is None:
        for a in ax[1]:
            a.text(0.5, 0.5, "N / A", ha="center", va="center",
                   fontsize=14, color="white")
            a.set_facecolor("black"); a.axis("off")
    else:
        for a, im, ttl in zip(ax[1, :2], [q_gt, q_rec], ["GT | QSM", "Rec | QSM"]):
            a.imshow(im, cmap=qsm_cmap, vmin=-qsm_max, vmax=qsm_max)
            a.set_title(ttl, fontsize=10); a.axis("off")

        # QSM Error (overlay)
        a_qerr = ax[1, 2]
        a_qerr.imshow(q_rec, cmap=qsm_cmap, vmin=-qsm_max, vmax=qsm_max)
        a_qerr.imshow(q_err, cmap=err_cmap, vmin=0, vmax=err_max, alpha=overlay_alpha)
        a_qerr.set_title("Err | QSM", fontsize=10); a_qerr.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------
# 에폭-단위 샘플 로깅
#   reconstructions, targets : validate() 가 반환한 dict[fname] = (S,H,W)
#   max_per_cat              : 카테고리(knee_x4 등) 당 몇 장까지 올릴지
# ---------------------------------------------------------------------
def log_epoch_samples(reconstructions, targets, step, max_per_cat=1):
    if not (wandb and wandb.run):
        return

    buckets = defaultdict(list)     # {cat: [wandb.Image, …]}

    for fname, vol in reconstructions.items():
        cat = _cat_from_fname(fname)
        if len(buckets[cat]) >= max_per_cat:
            continue

        mid = vol.shape[0] // 2
        fig = make_figure(
            np.abs(targets[fname][mid]), np.abs(vol[mid]),
            np.angle(targets[fname][mid]), np.angle(vol[mid]),
            f"{cat} | {fname} | slice {mid}"
        )
        buckets[cat].append(wandb.Image(fig))
        plt.close(fig)

    # ─── 한 번만 wandb.log ─────────────────────────────
    log_dict = {f"{cat}/samples": imgs for cat, imgs in buckets.items()}
    wandb.log(log_dict, step=step)

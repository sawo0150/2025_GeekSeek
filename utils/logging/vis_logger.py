# utils/wandb_tools/vis_logger.py
import numpy as np
import matplotlib.pyplot as plt
import wandb, re

def _cat_from_fname(fname):  # metric_accumulator와 동일
    organ = "knee" if "knee" in fname.lower() else "brain"
    acc   = "x4" if re.search(r"acc4|x4|r04", fname, re.I) else "x8"
    return f"{organ}_{acc}"

def make_figure(m_gt, m_rec, q_gt, q_rec, title=''):
    m_err, q_err = np.abs(m_gt-m_rec), np.abs(q_gt-q_rec)
    fig, ax = plt.subplots(2,3, figsize=(12,8))
    for a, (im, ttl) in zip(
        ax.ravel(),
        [(m_gt,'GT|Mag'),(m_rec,'Rec|Mag'),(m_err,'Err|Mag'),
         (q_gt,'GT|QSM'),(q_rec,'Rec|QSM'),(q_err,'Err|QSM')]):
        a.imshow(im, cmap='gray' if 'Mag' in ttl else 'turbo')
        a.set_title(ttl); a.axis('off')
    fig.suptitle(title); fig.tight_layout()
    return fig

def log_category_sample(fname, slice_idx, recon, target, step):
    """recon/target : (H,W) complex64 numpy"""
    cat = _cat_from_fname(fname)
    fig = make_figure(np.abs(target), np.abs(recon),
                      np.angle(target), np.angle(recon),
                      f"{cat} | {fname} | slice {slice_idx}")
    wandb.log({f"{cat}/sample": wandb.Image(fig)}, step=step)
    plt.close(fig)

"""
Hydra + W&B ���� ��ũ��Ʈ
FastMRI_challenge/main.py
"""
import hydra, wandb, torch, sys, os
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from types import SimpleNamespace

# repo ���� ��� ����Ʈ ��� Ȯ�� (train.py ��İ� ����) ������������������������������������������������
PROJECT_ROOT = Path(__file__).resolve().parent
for extra in ["utils/model", "utils/common"]:
    path = PROJECT_ROOT / extra
    if str(path) not in sys.path:
        sys.path.insert(1, str(path))

from utils.learning.train_part import train      # ���� �н� ����
from utils.common.utils import seed_fix          # seed ���� �Լ�

# ��������������������������������������������������������������������������������������������������������������������������������������������������������������
def _flatten_cfg_to_args(cfg: DictConfig) -> SimpleNamespace:
    """
    Hydra cfg �� ���� train.py���� ���� args ���ӽ����̽��� ��ȯ
    (train_part.py�� �״�ζ� �ʼ�)
    """
    flat = OmegaConf.to_container(cfg, resolve=True)
    args = SimpleNamespace()

    # 1) �ֻ��� Ű
    for k, v in flat.items():
        if k in {"model", "data", "wandb"}:
            continue
        setattr(args, k, v)

    # 2) model / data ���� Ű�� args�� ����
    for sub in ("model", "data"):
        for k, v in flat[sub].items():
            setattr(args, k, v)

    # 3) train_part.py�� ����ϴ� �빮�� �ʵ带 ������
    args.GPU_NUM = flat["GPU_NUM"]

    # 4) ��� ��� ���� (train.py ���� �ݿ�) :contentReference[oaicite:1]{index=1}
    result_dir = PROJECT_ROOT / "result" / args.net_name
    args.exp_dir = result_dir / "checkpoints"
    args.val_dir = result_dir / "reconstructions_val"
    args.main_dir = result_dir / Path(__file__).name
    args.val_loss_dir = result_dir
    for p in [args.exp_dir, args.val_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return args


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    # ���� 1. reproducibility ��������������������������������������������������������������������������������������������������������
    if cfg.seed is not None:
        seed_fix(cfg.seed)

    # ���� 2. cfg �� args ��ȯ -----------------------------------------------------
    args = _flatten_cfg_to_args(cfg)

    # ���� 3. W&B �ʱ�ȭ ----------------------------------------------------------
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=args.net_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # (����) �� �׷����Ʈ �ڵ� �α�
    wandb.watch(log="all", log_freq=cfg.report_interval)

    # ���� 4. �н� ---------------------------------------------------------------
    train(args)   # utils.learning.train_part.train ȣ�� :contentReference[oaicite:2]{index=2}

    # ���� 5. ������ -------------------------------------------------------------
    wandb.finish()


if __name__ == "__main__":
    main()

"""
Hydra + W&B 진입 스크립트
FastMRI_challenge/main.py
"""
import hydra, wandb, torch, sys, os
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from types import SimpleNamespace

# repo 내부 모듈 임포트 경로 확보 (train.py 방식과 동일) ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
for extra in ["utils/model", "utils/common"]:
    path = PROJECT_ROOT / extra
    if str(path) not in sys.path:
        sys.path.insert(1, str(path))

from utils.learning.train_part import train      # 기존 학습 루프
from utils.common.utils import seed_fix          # seed 고정 함수

# ───────────────────────────────────────────────────────────────────────────────
def _flatten_cfg_to_args(cfg: DictConfig) -> SimpleNamespace:
    """
    Hydra cfg → 기존 train.py에서 쓰던 args 네임스페이스로 변환
    (train_part.py가 그대로라 필수)
    """
    flat = OmegaConf.to_container(cfg, resolve=True)
    args = SimpleNamespace()

    # 1) 최상위 키
    for k, v in flat.items():
        if k in {"model", "data", "wandb"}:
            continue
        setattr(args, k, v)

    # 2) model / data 하위 키를 args에 주입
    for sub in ("model", "data"):
        for k, v in flat[sub].items():
            setattr(args, k, v)

    # 3) train_part.py가 기대하는 대문자 필드를 맞춰줌
    args.GPU_NUM = flat["GPU_NUM"]
    args.use_wandb = cfg.wandb.use_wandb
    args.max_vis_per_cat = cfg.wandb.max_vis_per_cat   # epoch 마다 카테고리별 이미지 수 (0 → 안 올림)

    # 4) Path 변환: data_path_* 를 Path 객체로 변경하여 load_data 에서의 '/' 연산 오류 방지
    if hasattr(args, 'data_path_train'):
        args.data_path_train = Path(args.data_path_train)
    if hasattr(args, 'data_path_val'):
        args.data_path_val = Path(args.data_path_val)

    # 5) 결과 경로 세팅 (train.py 로직 반영) :contentReference[oaicite:1]{index=1}
    result_dir = Path(cfg.data.PROJECT_ROOT) / "result" / args.net_name
    args.exp_dir = result_dir / "checkpoints"
    args.val_dir = result_dir / "reconstructions_val"
    args.main_dir = result_dir / Path(__file__).name
    args.val_loss_dir = result_dir
    for p in [args.exp_dir, args.val_dir]:
        p.mkdir(parents=True, exist_ok=True)

    return args


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    # ── 1. reproducibility ────────────────────────────────────────────────────
    if cfg.seed is not None:
        seed_fix(cfg.seed)

    # ── 2. cfg → args 변환 -----------------------------------------------------
    args = _flatten_cfg_to_args(cfg)
    
    # ── 3. W&B 초기화 ----------------------------------------------------------
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=args.net_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # (선택) 모델 그래디언트 자동 로깅
    # wandb.watch(log="all", log_freq=cfg.report_interval)

    # ── 4. 학습 ---------------------------------------------------------------
    train(args)   # utils.learning.train_part.train 호출 :contentReference[oaicite:2]{index=2}

    # ── 5. 마무리 -------------------------------------------------------------
    wandb.finish()


if __name__ == "__main__":
    main()

"""
Hydra + W&B 진입 스크립트
FastMRI_challenge/main.py
"""
from omegaconf import DictConfig, OmegaConf
import math, operator
# 안전한 계산용 최소 환경만 넘겨 줌 (※ __builtins__ 비우기)
OmegaConf.register_new_resolver(
    "calc",
    lambda expr: eval(expr, {"__builtins__": {}, "math": math, "operator": operator.__dict__})
)

import hydra, wandb, torch, sys, os
from collections.abc import Mapping
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
    ✔️  *재귀적으로* 모든 항목을 평탄화하여 SimpleNamespace 로 변환한다.
       - model / data 서브트리는 “옛 argparser 스타일” 유지 → prefix 미부여
       - 그 외 서브트리는 `<tree>_<leaf>` 형식으로 충돌 방지
    """
    container = OmegaConf.to_container(cfg, resolve=True)
    args = SimpleNamespace()

    def recurse(prefix: str, node: Mapping):
        for k, v in node.items():
            new_key = f"{prefix}_{k}" if prefix else k

            # special-case: dict를 '통째로' 보존할 키들
            PRESERVE = {"model", "data", "LRscheduler", "LossFunction", 
                        "optimizer", "compressor", "collator", "sampler",
                        "evaluation", "early_stop", "maskDuplicate","maskAugment"
                        ,"aug"
                        }

            # (1) 보존용 속성 먼저 세팅
            if prefix == "" and k in PRESERVE and isinstance(v, Mapping):
                # args.LRscheduler = {...}  처럼 딕셔너리 그대로 유지
                setattr(args, k, v)
                # (2) 동시에 하위 키도 평탄화해 주기
                recurse("", v) if k in {"model", "data"} else recurse(k, v)
            elif isinstance(v, Mapping):
                recurse(new_key, v)        # 일반 dict: prefix 유지
            else:
                setattr(args, new_key, v)

    recurse("", container)

    # 추가로 main.py 레거시 필드도 맞춰줌
    args.GPU_NUM         = container["GPU_NUM"]
    args.use_wandb       = container["wandb"]["use_wandb"]
    args.max_vis_per_cat = container["wandb"]["max_vis_per_cat"]

    # 4) Path 변환: data_path_* 를 Path 객체로 변경하여 load_data 에서의 '/' 연산 오류 방지
    if hasattr(args, 'data_path_train'):
        args.data_path_train = Path(args.data_path_train)
    if hasattr(args, 'data_path_val'):
        args.data_path_val = Path(args.data_path_val)

    # 5) 결과 경로 세팅 (train.py 로직 반영) :contentReference[oaicite:1]{index=1}
    result_dir = Path(cfg.data.PROJECT_ROOT) / "result" / args.exp_name
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
        name=args.exp_name,
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

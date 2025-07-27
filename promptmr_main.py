# promptmr_main.py
"""
Hydra + W&B 진입 스크립트 for Prompt-MR
1. Domain Classifier 학습
2. Reconstruction Model (PromptVarNet) 학습
"""
from omegaconf import DictConfig, OmegaConf
import math, operator
OmegaConf.register_new_resolver(
    "calc",
    lambda expr: eval(expr, {"__builtins__": {}, "math": math, "operator": operator.__dict__})
)

import hydra, wandb, torch, sys, os
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

# repo 내부 모듈 임포트 경로 확보
PROJECT_ROOT = Path(__file__).resolve().parent
for extra in ["utils/model", "utils/common", "utils/learning", "utils/data"]:
    path = PROJECT_ROOT / extra
    if str(path) not in sys.path:
        sys.path.insert(1, str(path))

# 필요한 모듈 임포트
from utils.learning.classifier_train_part import train_classifier
from utils.learning.train_part import train as train_reconstructor
from utils.common.utils import seed_fix

# _flatten_cfg_to_args 함수는 이전 답변의 최종 버전과 동일하게 유지합니다.
def _flatten_cfg_to_args(cfg: DictConfig) -> SimpleNamespace:
    container = OmegaConf.to_container(cfg, resolve=True)
    args = SimpleNamespace()

    def recurse(prefix: str, node: Mapping):
        for k, v in node.items():
            new_key = f"{prefix}_{k}" if prefix else k
            PRESERVE = {"model", "data", "LRscheduler", "LossFunction", 
                        "optimizer", "compressor", "collator", "sampler",
                        "evaluation", "early_stop", "maskDuplicate", "maskAugment",
                        "aug", "centerCropPadding", "deepspeed", "classifier",
                        "classifier_training"
                        }
            if k == "grad_accum_scheduler" and isinstance(v, Mapping):
                setattr(args, new_key, v)
                recurse(new_key, v)
                continue
            if prefix == "" and k in PRESERVE and isinstance(v, Mapping):
                setattr(args, k, v)
                # classifier를 특별 취급하지 않는 것이 핵심입니다.
                recurse("", v) if k in {"model", "data"} else recurse(k, v)
            elif isinstance(v, Mapping):
                recurse(new_key, v)
            else:
                setattr(args, new_key, v)

    recurse("", container)

    args.GPU_NUM         = container["GPU_NUM"]
    args.use_wandb       = container["wandb"]["use_wandb"]
    args.max_vis_per_cat = container["wandb"]["max_vis_per_cat"]
    args.deepspeed = container.get("training", {}).get("deepspeed")

    if hasattr(args, 'data_path_train'):
        args.data_path_train = Path(args.data_path_train)
    if hasattr(args, 'data_path_val'):
        args.data_path_val     = Path(args.data_path_val)

    result_dir = Path(cfg.data.PROJECT_ROOT) / "result" / args.exp_name
    args.exp_dir = result_dir / "checkpoints"
    args.val_dir = result_dir / "reconstructions_val"
    args.main_dir = result_dir / Path(__file__).name
    args.val_loss_dir = result_dir
    for p in [args.exp_dir, args.val_dir]:
        p.mkdir(parents=True, exist_ok=True)

    if "data" in cfg and "domain_filter" in cfg["data"]:
        args.domain_filter = cfg.data.domain_filter

    return args


@hydra.main(config_path="configs", config_name="train_promptmr", version_base=None)
def main(cfg: DictConfig):
    # 1. 재현성 설정
    if cfg.seed is not None:
        seed_fix(cfg.seed)

    # 2. 설정(cfg)을 args 객체로 변환
    args = _flatten_cfg_to_args(cfg)

    # 3. W&B 초기화
    if args.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=args.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # =========================================================================
    # [PROMPT-MR] 파이프라인 시작
    # =========================================================================

    # 4. 분류기 학습
    # [FIX] classifier_model을 여기서 인스턴스화하지 않습니다.
    # [FIX] train_classifier 함수에 args 객체 하나만 전달하여, 호출과 정의를 일치시킵니다.
    trained_classifier = train_classifier(args)
    trained_classifier.eval() # 평가 모드로 전환

    # 5. 재구성 모델 학습
    print("\n" + "="*80)
    print("PHASE 2: Training Reconstruction Model (PromptVarNet)")
    print("="*80)
    # train_reconstructor 함수에 학습된 분류기 객체를 전달
    train_reconstructor(args, classifier=trained_classifier)

    # 6. 마무리
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()

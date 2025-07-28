#!/usr/bin/env python
# resume_main.py

import sys
import wandb
from pathlib import Path

# ── main.py 과 동일한 import 경로 설정 (가장 먼저) ────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for extra in ["utils/model", "utils/common"]:
    p = PROJECT_ROOT / extra
    if str(p) not in sys.path:
        sys.path.insert(1, str(p))
# ────────────────────────────────────────────────────────────

import torch
from types import SimpleNamespace
from utils.learning.final_train_part import train   # 이제 경로가 통과됨
from utils.common.utils import seed_fix
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', '-c', required=True,
                   help='불러올 체크포인트(.pt) 경로')
    p.add_argument('--override_lr',    type=float, default=None,
                   help='학습률 재설정 (선택)')
    p.add_argument('--override_epochs',type=int,   default=None,
                   help='총 num_epochs 재설정 (선택)')
    return p.parse_args()

def main():
    args_cli = parse_args()
    ckpt_path = Path(args_cli.checkpoint)
    ckpt = torch.load(
        ckpt_path,
        map_location='cpu',
        weights_only=False    # <-- 전체 객체 로드 허용
    )


    # (1) 저장된 args(SimpleNamespace) 꺼내오기
    args = ckpt['args']


     # ── W&B 초기화 ───────────────────────────────────────────────
    if getattr(args, 'use_wandb', False):
        # args.wandb_project, args.wandb_entity, args.exp_name 은
        # _flatten_cfg_to_args() 로 만들어진 속성들입니다.
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{args.exp_name}_resume",
            config=vars(args)    # 또는 필요에 맞게 설정
        )
     # ──────────────────────────────────────────────────────────────
     
    # (2) Seed 고정 (원본과 동일하게)
    if getattr(args, 'seed', None) is not None:
        seed_fix(args.seed)

    # (3) 필요시 override
    if args_cli.override_lr is not None:
        args.lr = args_cli.override_lr
    if args_cli.override_epochs is not None:
        args.num_epochs = args_cli.override_epochs

    # (4) resume_checkpoint 경로 지정
    args.resume_checkpoint = str(ckpt_path)

    # (5) 실제 train() 호출
    train(args)

     # ── W&B 종료 ───────────────────────────────────────────────
    if getattr(args, 'use_wandb', False):
        wandb.finish()

     # ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()

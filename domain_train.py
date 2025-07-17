#!/usr/bin/env python3
"""
라운드-로빈 방식으로 도메인 전용 모델들을 번갈아 학습-검증-평가한다.

사용 예:
  python domain_train.py \
    --domain-groups '[["knee_x4","knee_x8"],["brain_x4"],["brain_x8"]]' \
    --config-paths 'configs/knee.yaml,configs/brain_x4.yaml,configs/brain_x8.yaml' \
    --epochs-per-block 5 \
    --total-epochs 50
"""
import argparse, json, math, os, sys
from pathlib import Path
from omegaconf import OmegaConf
import torch, wandb
from hydra import initialize, compose

# main.py 의 헬퍼 & train 루틴 재사용 (Hydra 데코레이터 없이)
from main import _flatten_cfg_to_args
from utils.learning.train_part import train

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain-groups", required=True,
                   help="JSON 리스트. 각 내부 리스트가 한 모델이 맡을 cat 집합")
    p.add_argument("--config-paths",  required=True,
                   help="쉼표구분. 모델별 베이스 YAML 경로들")
    p.add_argument("--epochs-per-block", type=int, default=5)
    p.add_argument("--total-epochs",     type=int, default=50)
    p.add_argument("--wandb-project",    default="fastmri_moe")
    return p.parse_args()

def run_block(cfg, start_epoch, end_epoch, project):
    """
    cfg: OmegaConf (이미 domain_filter, exp_name 등이 주입됨)
    start_epoch, end_epoch: 0-base inclusive/exclusive
    """
    # cfg.num_epochs = end_epoch
    # cfg.wandb.project = project

    # wandb_run_name = f"{cfg.exp_name}_e{start_epoch}-{end_epoch}"
    # cfg.wandb.name = wandb_run_name

    # # W&B 새 run
    # wandb.init(project=project,
    #            entity=cfg.wandb.entity,
    #            name=wandb_run_name,
    #            config=OmegaConf.to_container(cfg, resolve=True))
    
    # (1) 파라미터만 config에 패치
    cfg.num_epochs   = end_epoch
    cfg.wandb.project = project

    # (2) W&B run 이름은 init() 호출 시에만 지정
    run_name = f"{cfg.exp_name}_e{start_epoch}-{end_epoch}"
    wandb.init(
        project=project,
        entity=cfg.wandb.entity,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # 학습
    args_ns = _flatten_cfg_to_args(cfg)
    train(args_ns)

    wandb.finish()

def main():
    args = parse_args()
    domain_groups = json.loads(args.domain_groups)      # list[list[str]]
    cfg_paths      = args.config_paths.split(",")       # list[str]

    assert len(domain_groups) == len(cfg_paths), \
        "domain-groups 와 config-paths 개수가 달라요!"

    # 모델별 상태 테이블
    state = []
    for i,(doms,cfg_p) in enumerate(zip(domain_groups, cfg_paths)):
        work_dir = Path(f"result/domain{i}")
        work_dir.mkdir(parents=True, exist_ok=True)
        state.append(dict(cur_ep=0,
                          ckpt=work_dir/"checkpoints/model.pt",
                          doms=doms,
                          cfg_path=cfg_p,
                          work_dir=work_dir))

    while not all(s["cur_ep"] >= args.total_epochs for s in state):
        for i,s in enumerate(state):
            if s["cur_ep"] >= args.total_epochs:
                continue                      # 이미 끝난 모델

            run_epochs = min(args.epochs_per_block,
                             args.total_epochs - s["cur_ep"])
            
            # 1) 도메인별 exp_name 과 override 리스트 만들기
            doms        = s["doms"]
            start_ep    = s["cur_ep"]
            end_ep      = start_ep + run_epochs
            exp_name    = f"domain{i}_{'_'.join(doms)}"
            overrides   = [
                f"exp_name={exp_name}",
                f"+data.domain_filter=[{','.join(doms)}]",
                f"num_epochs={end_ep}",
                f"wandb.project={args.wandb_project}",
            ]
            if s["ckpt"].exists():
                overrides.append(f"resume_checkpoint={s['ckpt']}")

            # 2) Hydra 로 실제 컴포지션 수행
            with initialize(config_path="configs", job_name="domain_train", version_base=None):
                cfg = compose(
                    config_name="train",
                    overrides=overrides,
                )
            # 2) Hydra 로 실제 컴포지션 수행: config_paths 로 받은 파일(stem)을 config_name으로 사용
            cfg_file = s["cfg_path"]                  # e.g. "configs/domain.yaml"
            cfg_name = Path(cfg_file).stem            # → "domain"
            with initialize(config_path="configs", version_base=None):
                cfg = compose(
                    config_name=cfg_name,
                    overrides=overrides,
                )

            # 로그 확인용
            print(f"\n=== Domain{i} {s['doms']}  epochs {start_ep} → {end_ep} ===")
            run_block(cfg, start_ep, end_ep, args.wandb_project)

            s["cur_ep"] = end_ep               # 진척
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

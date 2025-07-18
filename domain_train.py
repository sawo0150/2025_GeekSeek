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
import argparse, json, math, os, sys, time
from pathlib import Path
from omegaconf import OmegaConf
# import torch, wandb
from hydra import initialize, compose
import subprocess, shlex

import os, matplotlib
os.environ["MPLBACKEND"] = "Agg"   # or matplotlib.use('Agg', force=True)

# main.py 의 헬퍼 & train 루틴 재사용 (Hydra 데코레이터 없이)
# from main import _flatten_cfg_to_args
# from utils.learning.train_part import train
# 문제의 원인: 부모 프로세스의 CUDA 컨텍스트
# 부모 프로세스의 GPU 점유: domain_train.py 스크립트에서 import torch를 하고 torch.cuda.empty_cache() 같은 함수를 호출하는 순간, 이 부모 파이썬 프로세스 자체에 CUDA 컨텍스트(Context)가 생성됩니다. 이 컨텍스트는 수백 MB의 GPU 메모리를 미리 점유합니다.

# 첫 번째 서브프로세스 실행: 첫 번째 main.py(knee)가 서브프로세스로 실행됩니다. 이때는 부모 프로세스가 점유한 메모리 외에 가용 메모리가 충분하므로 학습이 성공적으로 끝납니다.

# 메모리 해제 착시: 첫 번째 서브프로세스가 종료되면 해당 프로세스가 사용하던 GPU 메모리는 운영체제와 CUDA 드라이버에 의해 해제됩니다. 그래서 nvidia-smi로 보면 메모리가 거의 다 비어있는 것처럼 보입니다. (실제로 11464 MiB가 비어있다고 나왔죠.) 하지만 부모 프로세스가 점유한 CUDA 컨텍스트는 그대로 남아있습니다.

# 두 번째 서브프로세스 실행 및 OOM: 두 번째 main.py(brain)가 실행될 때, 첫 번째 실행 때와 미세하게 다른 메모리 상태 (부모 컨텍스트 + 드라이버의 미묘한 상태 차이)에서 시작합니다. 이 작은 차이가 메모리 단편화(fragmentation)를 유발하거나, 필요한 연속된 메모리 공간을 할당받지 못하게 만들어 torch.OutOfMemoryError를 발생시킵니다. 트레이스백을 보면 75.06 MiB is free라고 나오는데, 이는 두 번째 프로세스가 실행되는 시점에는 이미 메모리가 거의 꽉 찼다는 뜻입니다.


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--domain-groups", required=True,
                   help="JSON 리스트. 각 내부 리스트가 한 모델이 맡을 cat 집합")
    p.add_argument("--config-paths",  required=True,
                   help="쉼표구분. 모델별 베이스 YAML 경로들")
    p.add_argument("--epochs-per-block", type=int, default=5)
    p.add_argument("--total-epochs",     type=int, default=50)
    p.add_argument("--wandb-project",    default="fastmri_moe")
    p.add_argument("--result-root",      default="result",
                   help="Base directory for all experiments (will contain <exp_name>/checkpoints/model.pt)")

    return p.parse_args()


# subprocess 로 main.py 를 띄워 완전 격리 실행
def run_block_subprocess(cfg_name, overrides, start_ep, end_ep, project, resume_ckpt=None):

    # Hydra 는 `overrides` 만 인자로 받습니다. CLI 커스텀 플래그는 붙이면 안 됩니다.
    ov_cli = " ".join(shlex.quote(o) for o in overrides)   # 안전하게 quoting
    

    if start_ep == 0:
        # ── 처음 블록(처음 학습) ─────────────────────────
        cmd = f"python main.py --config-name {cfg_name} {ov_cli}"
    else:
        # ── 재개 블록(resume) ────────────────────────────
        # resume_ckpt 인자로 넘어온 체크포인트 경로를 --checkpoint 로 넘겨준다.
        if resume_ckpt is None:
            raise RuntimeError(f"재개할 체크포인트가 없습니다. start_ep={start_ep}")
        cmd = (
            f"python resume_main.py "
            f"-c {shlex.quote(str(resume_ckpt))} "
            f"--override_epochs {end_ep}"
        )


    print(f"Executing command: {cmd}") # 디버깅을 위해 실행될 명령어 출력
    subprocess.run(shlex.split(cmd), check=True)
    
def main():
    args = parse_args()
    domain_groups = json.loads(args.domain_groups)      # list[list[str]]
    cfg_paths      = args.config_paths.split(",")       # list[str]

    assert len(domain_groups) == len(cfg_paths), \
        "domain-groups 와 config-paths 개수가 달라요!"

    # # 모델별 상태 테이블
    # state = []
    # for i,(doms,cfg_p) in enumerate(zip(domain_groups, cfg_paths)):
    #     work_dir = Path(f"result/domain{i}")
    #     work_dir.mkdir(parents=True, exist_ok=True)
    #     state.append(dict(cur_ep=0,
    #                       ckpt=work_dir/"checkpoints/model.pt",
    #                       doms=doms,
    #                       cfg_path=cfg_p,
    #                       work_dir=work_dir))
        
    # 모델별 상태 테이블: exp_dir, ckpt 경로를 train_part의 저장 위치와 일치시킵니다
    state = []
    # for i,(doms,cfg_p) in enumerate(zip(domain_groups, cfg_paths)):
    #     exp_name = f"domain{i}_{'_'.join(doms)}"
    #     exp_dir  = Path("result")/exp_name
    #     exp_dir.mkdir(parents=True, exist_ok=True)
    #     state.append(dict(
    #         cur_ep  = 0,
    #         exp_dir = exp_dir,
    #         ckpt    = exp_dir/"model.pt",   # train_part.py 와 동일
    #         doms    = doms,
    #         cfg_path= cfg_p,
    #     ))
    for i,(doms,cfg_p) in enumerate(zip(domain_groups, cfg_paths)):
        exp_name = f"domain{i}_{'_'.join(doms)}"
        exp_dir  = Path(args.result_root)/exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        # ── checkpoint 은 exp_dir/checkpoints 에 저장된다고 가정 ──
        ckpt_dir = exp_dir/"checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state.append(dict(
           cur_ep  = 0,
            exp_dir = exp_dir,
            ckpt    = ckpt_dir/"model.pt",
            doms    = doms,
            cfg_path= cfg_p,
        ))

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
                f"+exp_dir={s['exp_dir']}",   # ✔️ exp_dir override 추가
            ]
            
            if s["ckpt"].exists():
                overrides.append(f"+resume_checkpoint={s['ckpt']}")

            # 프로세스 격리 실행: main.py 를 새 프로세스로 실행합니다.
            cfg_file = s["cfg_path"]
            cfg_name = Path(cfg_file).stem
            print(f"\n=== Launch subprocess for Domain{i} {s['doms']} epochs {start_ep}→{end_ep} ===")
            # run_block_subprocess(cfg_name, overrides, start_ep, end_ep, args.wandb_project)
            # s["ckpt"] 가 존재하면 resume_ckpt 에 넘겨주고, 아니면 None
            resume_ckpt = s["ckpt"] if s["ckpt"].exists() else None
            # print(resume_ckpt, s['ckpt'])
            run_block_subprocess(
                cfg_name,
                overrides,
                start_ep,
                end_ep,
                args.wandb_project,
                resume_ckpt=resume_ckpt
            )

            s["cur_ep"] = end_ep               # 진척
            # torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

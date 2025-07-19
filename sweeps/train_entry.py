
import argparse, subprocess, sys, os


def parse_args():
    parser = argparse.ArgumentParser()
    # ===== sweep 에서 넘길 파라미터들 =====
    parser.add_argument("--epoch", type=int, required=False)
    parser.add_argument("--maskDuplicate",  type=str)
    # mixed-precision sweep을 위해 amp 플래그 추가
    parser.add_argument("--amp", type=str, required=False,
                        help="mixed-precision on/off (true/false)")
    # 기존 optimizer
    parser.add_argument("--optimizer",    type=str, required=False)
    # 추가로 sweep 할 파라미터들
    parser.add_argument("--loss_function", type=str, required=False)
    parser.add_argument("--mask_only",     type=str, required=False)
    parser.add_argument("--region_weight", type=str, required=False)
    # 우리가 실험할 항목
    parser.add_argument("--scheduler")                # Cosine…, StepLR …+    
    # 모델 그룹 선택 (Hydra: model=...)
    parser.add_argument("--model",         type=str, required=False,
                        help="which model config to use (e.g. fivarnet, ifvarnet, featurevarnet_sh_w, ...)")

    # Hydra 기본 config-name도 뽑아두기
    parser.add_argument("--config-name",   type=str, dest="config_name", required=False)
    return parser.parse_known_args()

args, unknown = parse_args()


cmd = ["python", "main.py"]

# 1) --config-name 은 Hydra 쪽으로
if args.config_name:
    cmd.append(f"--config-name={args.config_name}")

# 2) maskDuplicate 단일 파라미터 sweep
if args.maskDuplicate:
    cmd.append(f"maskDuplicate={args.maskDuplicate}")
    if args.maskDuplicate == "acc4_acc8":
        cmd.append(f"num_epochs={30}")     # ⚡️ 추가
    else:
        cmd.append(f"num_epochs={60}")     # ⚡️ 추가



# 2) optimizer override
if args.optimizer:
    cmd.append(f"optimizer={args.optimizer}")

if args.epoch is not None:
    cmd.append(f"num_epochs={args.epoch}")     # ⚡️ 추가

# 3) LossFunction 그룹 선택 및 그룹 내 파라미터 override
if args.loss_function:
    # (a) 그룹 이름 선택
    cmd.append(f"LossFunction={args.loss_function}")
    # (b) 그룹 내 옵션 override
    if args.mask_only is not None:
        cmd.append(f"LossFunction.mask_only={args.mask_only}")
    if args.region_weight is not None:
        cmd.append(f"LossFunction.region_weight={args.region_weight}")

# ───────────────── LRscheduler 처리 ─────────────────
if args.scheduler:
    cmd.append(f"LRscheduler={args.scheduler}")
    
# ─────────────────── amp 처리 ────────────────────
if args.amp is not None:
    # Hydra group 'training' 내부의 amp 옵션을 override
    cmd.append(f"training.amp={args.amp}")

# ────────────────── model override ──────────────────
if args.model:
    # Hydra 모델 그룹을 override
    cmd.append(f"model={args.model}")
 
# 4) 나머지 unknown 은 그대로 (다른 Hydra 플래그 있으면 받기)
for arg in unknown:
    if arg.startswith('--'):
        cmd.append(arg[2:]) # 앞의 '--' 두 글자 제거
    else:
        cmd.append(arg)
        
sys.exit(subprocess.call(cmd))
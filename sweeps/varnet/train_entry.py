import argparse, subprocess, sys, os

def parse_args():
    parser = argparse.ArgumentParser()
    
    # ===== W&B sweep에서 넘길 파라미터들 =====
    # ⭐️ VarNet 모델 스윕을 위해 --model 인수 추가
    parser.add_argument("--model", type=str, required=False, help="Hydra의 model config 그룹을 선택합니다.")
    
    # --- 기존에 사용되던 다른 인수들 ---
    parser.add_argument("--epoch", type=int, required=False)
    parser.add_argument("--maskDuplicate",  type=str)
    parser.add_argument("--amp", type=str, required=False, help="mixed-precision on/off (true/false)")
    parser.add_argument("--optimizer",    type=str, required=False)
    parser.add_argument("--loss_function", type=str, required=False)
    parser.add_argument("--mask_only",     type=str, required=False)
    parser.add_argument("--region_weight", type=str, required=False)
    parser.add_argument("--scheduler", type=str)
    parser.add_argument("--config-name",   type=str, dest="config_name", required=False)
    
    return parser.parse_known_args()

# W&B agent로부터 받은 인수를 파싱합니다.
args, unknown = parse_args()

# 최종적으로 실행할 명령어 리스트를 생성합니다.
cmd = ["python", "main.py"]

# 1) --config-name 은 Hydra 쪽으로 전달
if args.config_name:
    cmd.append(f"--config-name={args.config_name}")

# ⭐️ 추가: --model 인수가 있으면 모델 그룹과 실험 이름을 override
if args.model:
    # Hydra가 model config를 선택하도록 'model=모델이름' 형식으로 추가
    cmd.append(f"model={args.model}")
    # W&B에서 실행 이름을 명확히 구분하기 위해 exp_name도 동적으로 설정
    cmd.append(f"exp_name=varnet_sweep_{args.model}")

# 2) 다른 Hydra 파라미터 override 처리
if args.maskDuplicate:
    cmd.append(f"maskDuplicate={args.maskDuplicate}")
    cmd.append(f"num_epochs={30 if args.maskDuplicate == 'acc4_acc8' else 60}")

if args.optimizer:
    cmd.append(f"optimizer={args.optimizer}")

if args.epoch is not None:
    cmd.append(f"num_epochs={args.epoch}")

if args.loss_function:
    cmd.append(f"LossFunction={args.loss_function}")
    if args.mask_only is not None:
        cmd.append(f"LossFunction.mask_only={args.mask_only}")
    if args.region_weight is not None:
        cmd.append(f"LossFunction.region_weight={args.region_weight}")

if args.scheduler:
    cmd.append(f"LRscheduler={args.scheduler}")
    
if args.amp is not None:
    cmd.append(f"training.amp={args.amp}")

# 3) W&B가 넘겨준 나머지 unknown 인수를 그대로 Hydra에 전달
for arg in unknown:
    if arg.startswith('--'):
        cmd.append(arg[2:]) # '--' 제거
    else:
        cmd.append(arg)

# 생성된 명령어로 main.py를 실행합니다.
print(f"Executing command: {' '.join(cmd)}")
sys.exit(subprocess.call(cmd))
# sweeps/mraugment/train_entry.py

import argparse, subprocess, sys, os

def parse_args():
    """
    W&B의 command 섹션으로부터 --key=value 형태의 인수를 받도록 정의합니다.
    """
    parser = argparse.ArgumentParser()

    # --- Hydra 기본 인자 ---
    parser.add_argument("--config-name", type=str, required=False)

    # --- Augmentation 전체 제어 ---
    parser.add_argument("--aug", type=str, required=False)

    # --- MRAugmenter 스케줄 관련 인자 ---
    parser.add_argument("--aug_schedule_mode", type=str, required=False)
    parser.add_argument("--aug_schedule_type", type=str, required=False)
    parser.add_argument("--aug_exp_decay", type=float, required=False)
    parser.add_argument("--aug_strength", type=float, required=False)

    # --- MRAugmenter weight_dict 관련 인자 (wd_ prefix 사용) ---
    parser.add_argument("--wd_fliph", type=float, required=False)
    parser.add_argument("--wd_flipv", type=float, required=False)
    parser.add_argument("--wd_rotate", type=float, required=False)
    parser.add_argument("--wd_scale", type=float, required=False)
    parser.add_argument("--wd_shift", type=float, required=False)
    parser.add_argument("--wd_shear", type=float, required=False)

    # --- 기타 학습 관련 인자 ---
    parser.add_argument("--epoch", type=int, required=False)
    parser.add_argument("--maskDuplicate", type=str, required=False)

    return parser.parse_known_args()

# 1. W&B agent의 `command:`로부터 인수를 파싱합니다.
args, unknown = parse_args()

# 2. 최종적으로 Hydra(main.py)에 전달할 명령어를 새로 생성합니다.
cmd = ["python", "main.py"]

# 3. 파싱된 인수를 Hydra가 이해하는 형식으로 "번역" 및 "재구성"합니다.

# (A) Hydra 특별 명령어(--config-name)는 "--"를 유지합니다.
if args.config_name:
    cmd.append(f"--config-name={args.config_name}")

# (B) 값 오버라이드는 "--"를 제거하고, 필요한 경우 키 이름을 변경합니다.
if args.aug is not None:
    cmd.append(f"aug={args.aug}")
if args.aug_schedule_mode is not None:
    cmd.append(f"aug.aug_schedule_mode={args.aug_schedule_mode}")
if args.aug_schedule_type is not None:
    cmd.append(f"aug.aug_schedule_type={args.aug_schedule_type}")
if args.aug_exp_decay is not None:
    cmd.append(f"aug.aug_exp_decay={args.aug_exp_decay}")
if args.aug_strength is not None:
    cmd.append(f"aug.aug_strength={args.aug_strength}")

if args.wd_fliph is not None:
    cmd.append(f"aug.weight_dict.fliph={args.wd_fliph}")
if args.wd_flipv is not None:
    cmd.append(f"aug.weight_dict.flipv={args.wd_flipv}")
if args.wd_rotate is not None:
    cmd.append(f"aug.weight_dict.rotate={args.wd_rotate}")
if args.wd_scale is not None:
    cmd.append(f"aug.weight_dict.scale={args.wd_scale}")
if args.wd_shift is not None:
    cmd.append(f"aug.weight_dict.shift={args.wd_shift}")
if args.wd_shear is not None:
    cmd.append(f"aug.weight_dict.shear={args.wd_shear}")

if args.epoch is not None:
    cmd.append(f"num_epochs={args.epoch}") # 'epoch' -> 'num_epochs'로 번역
if args.maskDuplicate is not None:
    cmd.append(f"maskDuplicate={args.maskDuplicate}")

# (C) 혹시 모를 나머지 인수를 안전하게 처리합니다.
for arg in unknown:
    if arg.startswith('--') and '=' in arg:
        cmd.append(arg[2:])
    else:
        cmd.append(arg)

# 4. 최종 생성된 명령어를 출력하고 실행합니다.
print(f"Executing command: {' '.join(cmd)}")
sys.exit(subprocess.call(cmd))
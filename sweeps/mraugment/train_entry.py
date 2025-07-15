# sweeps/mraugment/train_entry.py

import argparse, subprocess, sys, os

def parse_args():
    parser = argparse.ArgumentParser()

    # --- Hydra 기본 인자 ---
    parser.add_argument("--config-name", type=str, dest="config_name", required=False)

    # --- Augmentation 전체 제어 ---
    parser.add_argument("--aug", type=str, required=False, help="e.g., 'none' or 'mraugment'")

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
    
    # +++ 수정된 부분 시작 +++
    # maskDuplicate 기능을 제어하기 위한 인자 추가
    parser.add_argument("--maskDuplicate", type=str, required=False, help="e.g., 'none' or 'acc4_acc8'")
    # +++ 수정된 부분 끝 +++

    return parser.parse_known_args()

args, unknown = parse_args()

# 기본 실행 명령어
cmd = ["python", "main.py"]

# 1) --config-name 은 Hydra 쪽으로
if args.config_name:
    cmd.append(f"--config-name={args.config_name}")

# 2) Augmentation 관련 파라미터 추가
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

# 3) Weight Dict 관련 파라미터 추가
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

# 4) 기타 파라미터
if args.epoch is not None:
    cmd.append(f"num_epochs={args.epoch}")

# +++ 수정된 부분 시작 +++
# 5) maskDuplicate 파라미터 추가
if args.maskDuplicate is not None:
    cmd.append(f"maskDuplicate={args.maskDuplicate}")
# +++ 수정된 부분 끝 +++


# 6) 나머지 unknown 은 그대로 전달 (다른 Hydra 플래그용)
cmd.extend(unknown)

sys.exit(subprocess.call(cmd))
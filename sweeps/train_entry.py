
import argparse, subprocess, sys, os


def parse_args():
    parser = argparse.ArgumentParser()
    # 기존 optimizer
    parser.add_argument("--optimizer",    type=str, required=False)
    # 추가로 sweep 할 파라미터들
    parser.add_argument("--loss_function", type=str, required=False)
    parser.add_argument("--mask_only",     type=str, required=False)
    parser.add_argument("--region_weight", type=str, required=False)
    # Hydra 기본 config-name도 뽑아두기
    parser.add_argument("--config-name",   type=str, dest="config_name", required=False)
    return parser.parse_known_args()

args, unknown = parse_args()


cmd = ["python", "main.py"]

# 1) --config-name 은 Hydra 쪽으로
if args.config_name:
    cmd.append(f"--config-name={args.config_name}")

# 2) optimizer override
if args.optimizer:
    cmd.append(f"optimizer={args.optimizer}")

# 3) LossFunction 그룹 선택 및 그룹 내 파라미터 override
if args.loss_function:
    # (a) 그룹 이름 선택
    cmd.append(f"LossFunction={args.loss_function}")
    # (b) 그룹 내 옵션 override
    if args.mask_only is not None:
        cmd.append(f"LossFunction.mask_only={args.mask_only}")
    if args.region_weight is not None:
        cmd.append(f"LossFunction.region_weight={args.region_weight}")

# 4) 나머지 unknown 은 그대로 (다른 Hydra 플래그 있으면 받기)
cmd += unknown

sys.exit(subprocess.call(cmd))
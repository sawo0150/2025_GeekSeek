"""
W&B sweep entrypoint  ➜  Hydra main.py 로 파라미터 변환
"""
import argparse, subprocess, sys, os

parser = argparse.ArgumentParser()
parser.add_argument("--optimizer", type=str, required=True)
# 필요하면 다른 sweep 파라미터도 추가 (--lr 등)
args, unknown = parser.parse_known_args()

cmd = ["python", "main.py", f"optimizer={args.optimizer}"] + unknown
# unknown 리스트에는 Hydra override 형식 요소가 없을 테니 그대로 전달해도 무방
sys.exit(subprocess.call(cmd))

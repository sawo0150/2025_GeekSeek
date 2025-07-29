#!/usr/bin/env bash
# 두 번 연속 학습 스크립트 ─ exp_name만 run1·run2로 바꿔서 실행
set -e          # 중간에 오류 발생 시 즉시 중단
set -o pipefail # 파이프라인에서도 오류 전파

# ★ 필요하면 프로젝트 루트로 이동
# cd /path/to/FastMRI_challenge

BASE_EXP="ampXSeedFix_fm"      # train.yaml의 기본 exp_name
PYTHON_ENTRY="final_main.py"  # Hydra 진입 스크립트

for run in 1 2; do
  EXP_NAME="${BASE_EXP}_run${run}"
  echo "========== [${run}/2] 시작: exp_name=${EXP_NAME} =========="
  
  # Hydra override: exp_name만 교체
  python -u "${PYTHON_ENTRY}" exp_name="${EXP_NAME}"
  
  echo "========== [${run}/2] 종료: exp_name=${EXP_NAME} =========="
  echo
done
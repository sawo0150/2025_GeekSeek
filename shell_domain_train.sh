#!/usr/bin/env bash

# ✅ PyTorch CUDA 메모리 할당자 설정을 추가하여 단편화 문제 해결 시도
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 스크립트 실행 중 오류가 발생하면 즉시 중단
set -e

# --- 설정 변수 ---
TOTAL_EPOCHS=20
EPOCHS_PER_BLOCK=1
WANDB_PROJECT="fastmri_domain_train"

# --- 도메인 그룹 및 설정 파일 정의 ---
# 배열은 공백으로 요소를 구분합니다.
# 각 그룹은 "도메인1,도메인2" 형태의 문자열로 정의합니다.
declare -a DOMAIN_GROUPS=(
    "brain_x4,brain_x8"
    "knee_x4,knee_x8"
)
# 각 그룹에 해당하는 설정 파일 경로를 정의합니다.
declare -a CONFIG_NAMES=(
    "domain" # configs/domain.yaml 에 해당
    "domain" # configs/domain.yaml 에 해당 (예시이며, 필요시 변경)
)

# --- 모델별 현재 에포크 추적 ---
declare -a CURRENT_EPOCHS
for i in "${!DOMAIN_GROUPS[@]}"; do
    CURRENT_EPOCHS[$i]=0
done

# --- 메인 루프 ---
while true; do
    ALL_DONE=1 # 모든 작업이 완료되었는지 확인하는 플래그

    for i in "${!DOMAIN_GROUPS[@]}"; do
        # 현재 모델의 학습이 끝나지 않았다면 계속 진행
        if [ ${CURRENT_EPOCHS[$i]} -lt $TOTAL_EPOCHS ]; then
            ALL_DONE=0 # 아직 할 일이 남았음

            # 이번 블록에서 실행할 에포크 계산
            START_EP=${CURRENT_EPOCHS[$i]}
            END_EP=$((START_EP + EPOCHS_PER_BLOCK))
            if [ $END_EP -gt $TOTAL_EPOCHS ]; then
                END_EP=$TOTAL_EPOCHS
            fi

            DOMS=${DOMAIN_GROUPS[$i]}
            CFG_NAME=${CONFIG_NAMES[$i]}
            EXP_NAME="domain${i}_$(echo $DOMS | tr ',' '_')"
            CKPT_PATH="result/domain${i}/checkpoints/model.pt"
            
            echo ""
            echo "=== Launching Domain${i} [${DOMS}] epochs ${START_EP} -> ${END_EP} ==="

            # Override 리스트 구성
            OVERRIDES=(
                "exp_name=${EXP_NAME}"
                "+data.domain_filter=[${DOMS}]"
                "num_epochs=${END_EP}"
                "wandb.project=${WANDB_PROJECT}"
            )
            if [ -f "$CKPT_PATH" ]; then
                OVERRIDES+=("resume_checkpoint=${CKPT_PATH}")
            fi

            # 명령어 실행
            python main.py --config-name "$CFG_NAME" "${OVERRIDES[@]}"

            # 💥 작업 종료 후 잠시 대기하고 wandb 관련 프로세스 강제 종료
            echo "Waiting for 5 seconds and cleaning up lingering processes..."
            sleep 5
            pkill -f "wandb-core" || true # wandb-core 프로세스를 찾아 종료 (오류 무시)

            # 현재 에포크 업데이트
            CURRENT_EPOCHS[$i]=$END_EP
        fi
    done

    # 모든 모델의 학습이 끝났으면 루프 종료
    if [ $ALL_DONE -eq 1 ]; then
        echo "All domains have been trained for $TOTAL_EPOCHS epochs."
        break
    fi
done
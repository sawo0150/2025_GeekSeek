# FastMRI Challenge 2025 · Baby VarNet 리포지터리

> **한 줄 요약** — MRI 재구성 모델 연구를 위한 트레이닝∙인퍼런스 파이프라인에 Hydra (파라미터 관리)와 Weights & Biases (실험 트래킹)를 통합했습니다. YAML 오버라이드 한 줄이면 새 실험을 만들고, W\&B 대시보드에서 모든 로그를 실시간으로 볼 수 있습니다.

---

## 0. 주요 변경 사항 (Hydra & W\&B)

|  기능       |  기존 스크립트                      |  업데이트 후                                            |
| --------- | ----------------------------- | -------------------------------------------------- |
|  파라미터 관리  |  `train.py` 내 하드코딩 / bash 인자  |  `configs/*.yaml` 로 모델·데이터·학습 파라미터 모듈화             |
|  실험 로그    |  별도 로그 없음 + npy 저장            |  W\&B 대시보드 (스칼라·이미지·히스토그램 등)                       |
|  스윕 탐색    |  직접 for 문 스크립트 작성             |  `sweep.yaml` + `wandb sweep/agent`                |
|  새 모델 추가  |  train.py 수정 필요               |  `utils/model/__init__.py`의 `model_registry`에 등록만  |

---
dddd
## 1. 폴더 구조 (요약)

```
FastMRI_challenge/
├─ configs/        # Hydra YAML
│  ├─ train.yaml
│  ├─ model/
│  │   ├─ varnet_small.yaml
│  │   └─ varnet_large.yaml
│  └─ data/
│     ├─ local.yaml
│     └─ cluster.yaml
├─ main.py         # Hydra + W&B 진입점 ★NEW
├─ sweep.yaml      # W&B 스윕 설정 ★NEW
├─ utils/
│  ├─ model/
│  │   ├─ varnet.py
│  │   └─ …       # 새 모델 추가 가능
│  └─ …
└─ result/         # 자동 생성·저장 경로
```

### 1.1 Data / result 폴더 상세

기존 README 의 데이터·결과 디렉터리 설명은 변경 없으므로 [기존 문서](docs/…)를 참조하세요.

---

## 2. 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Python 3.9 \~ 3.12 권장**

---

## 3. 기본 학습 & 추론 명령어

```bash
# 학습 (기본 파라미터)
python main.py

# 리더보드 데이터 재구성
python reconstruct.py --model_path result/varnet_small/checkpoints/best_model.pt
```

### 3.1 하이퍼파라미터 오버라이드 예시

```bash
# 러닝레이트와 에폭 수 변경
python main.py lr=5e-4 num_epochs=20

# 모델 바꿔 돌리기
python main.py model=varnet_large batch_size=2
```

모든 파라미터 키는 `configs/train.yaml` 및 하위 YAML을 참고하세요.

---

## 4. W\&B 대시보드 사용법

* 실행 시 자동으로 프로젝트가 생성되고 실험마다 러닝 커브·SSIM·이미지 프리뷰가 업로드됩니다.
* 조직/프로젝트 경로는 `configs/train.yaml > wandb` 섹션에서 수정합니다.
* 로깅 추가 예시 (`utils/learning/train_part.py`):

  ```python
  wandb.log({
      "train_loss": loss.item(),
      "val_ssim": val_ssim,
      "lr": scheduler.get_last_lr()[0]
  }, step=global_step)
  ```

---

## 5. Sweep (하이퍼파라미터 탐색)

1. `sweep.yaml` 편집 → 탐색할 파라미터만 나열
2. 터미널에서 

```bash
wandb sweep sweep.yaml          # ID 발급
wandb agent <SWEEP_ID>         # 에이전트 실행
```

3. Hydra 오버라이드 규칙이 그대로 적용되므로, `model.cascade`처럼 하위 키도 직접 기재 가능.

---

## 6. 새 모델 추가 가이드

1. `utils/model/your_model.py`에 클래스 작성.
2. `utils/model/__init__.py`에서:

```python
from .utils.model.your_model import YourModel
model_registry["your_model"] = YourModel
```

3. `configs/model/your_model.yaml` 생성 후,

```bash
python main.py model=your_model
```

> **train\_part.py** 는 `args.net_name` 으로 모델을 선택하므로 추가 수정 필요 없습니다.

---

## 7. 로드맵 & 구현할 기능

* \[ ] Mixed‑Precision (AMP) 지원
* \[ ] 다중‑GPU / DDP 학습 모드
* \[ ] 자동 데이터 다운로드 스크립트

---


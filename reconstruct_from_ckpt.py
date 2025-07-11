# reconstruct_from_ckpt.py
"""
사용법 예
---------
$ python reconstruct_from_ckpt.py \
        --checkpoint result/small_varnet_resume_test/checkpoints/best_model.pt \
        --leaderboard_root /home/fastMRI/Data/leaderboard \
        --gpu 1 --batch_size 2
"""

import argparse, copy, sys, time, torch, os
from pathlib import Path
from types import SimpleNamespace

# ───────── 경로 삽입 & 충돌 제거 ──────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))            # ① 최우선 검색 경로

# ② 외부 utils 모듈이 이미 import돼 있으면 제거
if 'utils' in sys.modules and not hasattr(sys.modules['utils'], '__path__'):
    del sys.modules['utils']

# 필요한 서브폴더도 path 에 추가 (모델·공용 모듈)
for extra in ["utils/model", "utils/common"]:
    p = PROJECT_ROOT / extra
    if str(p) not in sys.path:
        sys.path.append(str(p))
# ────────────────────────────────────────

from utils.learning.test_part import forward as recon_forward

# ---------------------------------------------------------------------
def load_args_from_ckpt(ckpt_path: Path) -> SimpleNamespace:
    """checkpoint 속 args(SimpleNamespace) 를 꺼내어 리턴"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]

    # 간혹 dict 로 저장된 케이스 대비
    if not isinstance(args, SimpleNamespace):
        args = SimpleNamespace(**args)

    # exp_dir → 체크포인트 폴더 (model 가중치 로드에 필요)
    args.exp_dir = ckpt_path.parent
    # net_name (폴더 이름) → 편의상 추가
    args.net_name = getattr(args, "exp_name", ckpt_path.parent.parent.name)

    return args


def prepare_recon_args(
    base_args: SimpleNamespace,
    gpu: int,
    batch_size: int,
) -> SimpleNamespace:
    """GPU, batch_size 등 런타임 옵션만 덮어쓰기"""
    args = copy.deepcopy(base_args)
    args.GPU_NUM = gpu
    args.batch_size = batch_size
    return args


def reconstruct_for_acc(
    args_base: SimpleNamespace,
    acc_tag: str,
    leaderboard_root: Path,
):
    """acc4 / acc8 한 세트에 대해 forward() 실행"""
    args = copy.deepcopy(args_base)

    # ─ Leaderboard 데이터 경로 설정 ─
    args.data_path = leaderboard_root / acc_tag

    # ─ 결과 저장 폴더 (…/result/<exp_name>/reconstructions_leaderboard/accX) ─
    args.forward_dir = (
        args.exp_dir.parent / "reconstructions_leaderboard" / acc_tag
    )
    args.forward_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{acc_tag}] save → {args.forward_dir}")
    recon_forward(args)
    torch.cuda.empty_cache()  # 메모리 정리


# ---------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="Reconstruct leaderboard volumes from a saved checkpoint (no YAML)."
    )
    p.add_argument("-c", "--checkpoint", required=True, type=Path)
    p.add_argument("-p", "--leaderboard_root", type=Path, default="Data/leaderboard")
    p.add_argument("-g", "--gpu", type=int, default=0)
    p.add_argument("-b", "--batch_size", type=int, default=1)
    args_cli = p.parse_args()

    # 1) checkpoint → base args 복원
    base_args = load_args_from_ckpt(args_cli.checkpoint)

    # 2) 런타임 옵션 덮어쓰기
    base_args = prepare_recon_args(base_args, args_cli.gpu, args_cli.batch_size)
    # print(args_cli.batch_size)
    # print(args_cli.leaderboard_root)
    # 3) reconstruct acc4 / acc8
    t0 = time.time()
    for acc in ("acc4", "acc8"):
        reconstruct_for_acc(base_args, acc, args_cli.leaderboard_root)

    print(f"✅ Total reconstruction time : {(time.time()-t0):.1f} s")


if __name__ == "__main__":
    main()

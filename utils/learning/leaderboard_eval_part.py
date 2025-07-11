"""
통합 파이프라인에서 쓰기 편하도록
reconstruct.py · leaderboard_eval.py 의 'forward' 를 thin-wrapper 로 묶어둔다.
"""
# ---------------------------------------------------------------------------
#  FastMRI – leaderboard_eval_part.py
#  * acc4/acc8 재구성  →  SSIM 계산  →  dict 반환
#    - 1순위 : 새 `reconstruct_from_ckpt.py` 사용 (체크포인트 기반)
#    - 2순위 : 기존 `reconstruct.py` 의 forward(SimpleNamespace) 사용
# ---------------------------------------------------------------------------

from types import SimpleNamespace
from pathlib import Path
import torch, importlib, time
from tqdm import tqdm          # ★ progress bar

# --- reconstruct 모듈 선택 ---------------------------------------------------
# reconstruct_from_ckpt.py 가 있으면 우선 사용
try:
    rckpt_mod = importlib.import_module("reconstruct_from_ckpt")
    HAVE_RCKPT = True
except ModuleNotFoundError:
    HAVE_RCKPT = False

# fallback: 예전 방식의 reconstruct.py
recon_mod = importlib.import_module("reconstruct")   # 없으면 그대로 Exception

# leaderboard 평가 모듈 (고정)
eval_mod  = importlib.import_module("leaderboard_eval")


def run_leaderboard_eval(
        model_ckpt_dir: Path,
        leaderboard_root: Path,
        gpu: int = 0,
        batch_size: int = 1,
        output_key: str = "reconstruction",
    ):
    """
    ①  acc4/acc8 recon → ../reconstructions_leaderboard/<accX>
    ②  SSIM 계산 → (ssim4, ssim8, mean)
    """
    recon_root = model_ckpt_dir.parent / "reconstructions_leaderboard"
    recon_root.mkdir(parents=True, exist_ok=True)

    # ---------- 1. Reconstruction ----------
    if HAVE_RCKPT:
        # ① 체크포인트 선택 (best 우선 → model.pt)
        ckpt_path = (model_ckpt_dir / "best_model.pt") if (model_ckpt_dir / "best_model.pt").exists() \
                    else (model_ckpt_dir / "model.pt")

        # ② args 복원 + 런타임 옵션 적용
        base_args = rckpt_mod.load_args_from_ckpt(ckpt_path)
        base_args = rckpt_mod.prepare_recon_args(base_args, gpu, batch_size)

        # ③ acc4 / acc8 각각 reconstruct
        for acc in ("acc4", "acc8"):
            rckpt_mod.reconstruct_for_acc(base_args, acc, Path(leaderboard_root))
    else:
        # (구버전) reconstruct.py 의 forward(SimpleNamespace) 이용
        for acc in ("acc4", "acc8"):
            args_recon = SimpleNamespace(
                GPU_NUM=gpu,
                batch_size=batch_size,
                net_name=model_ckpt_dir.parent.name,
                path_data=Path(leaderboard_root) / acc,
                cascade=None, chans=None, sens_chans=None,  # 그대로 두면 VarNet default
                input_key="kspace",
                exp_dir=model_ckpt_dir,
                data_path=None, forward_dir=None,           # reconstruct.py 내부에서 쓰임
            )
            recon_mod.forward(args_recon)   # 저장만 하고 리턴값은 무시

    # ---------- 2. Evaluation ----------
    ssim = {}
    for acc in tqdm(("acc4", "acc8"), desc="SSIM-eval", leave=False):
        args_eval = SimpleNamespace(
            GPU_NUM=gpu,
            leaderboard_data_path=Path(leaderboard_root) / acc / "image",
            your_data_path=recon_root / acc,
            output_key=output_key,
        )
        ssim[acc] = eval_mod.forward(args_eval)

    ssim["mean"] = (ssim["acc4"] + ssim["acc8"]) / 2
    return ssim

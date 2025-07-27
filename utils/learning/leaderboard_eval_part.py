# utils/learning/leaderboard_eval_part.py
"""
[PROMPT-MR] 버전: 모델 객체를 직접 받아 리더보드 평가를 수행.
외부 스크립트 의존성을 제거하고, 현재 학습된 모델의 상태를 직접 평가.
"""
from types import SimpleNamespace
from pathlib import Path
import torch
import importlib
from tqdm import tqdm

from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions

try:
    eval_mod = importlib.import_module("leaderboard_eval")
except ImportError:
    print("Warning: 'leaderboard_eval' module not found. SSIM calculation will fail.")
    eval_mod = None

try:
    from utils.learning.test_part import test
except ImportError:
    print("Warning: 'test_part' module not found. Reconstruction will fail.")
    test = None


def run_leaderboard_eval(
        args: SimpleNamespace,
        model: torch.nn.Module,
        classifier: torch.nn.Module,
    ):
    """
    [PROMPT-MR] 버전: 모델 객체를 직접 받아 리더보드 평가를 수행
    ① acc4/acc8 recon -> ../reconstructions_leaderboard/<accX>
    ② SSIM 계산 -> (ssim4, ssim8, mean)
    """
    if not eval_mod or not test:
        print("Required modules for leaderboard evaluation are missing. Aborting.")
        return {"acc4": 0, "acc8": 0, "mean": 0}

    _raw_eval = getattr(args, "evaluation", {})
    eval_cfg = _raw_eval.get("evaluation", _raw_eval)

    leaderboard_root = Path(eval_cfg["leaderboard_root"])
    recon_root = Path(args.exp_dir).parent / "reconstructions_leaderboard"
    recon_root.mkdir(parents=True, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()
    if classifier:
        classifier.to(device)
        classifier.eval()

    # ---------- 1. Reconstruction ----------
    for acc in tqdm(("acc4", "acc8"), desc="Leaderboard-Recon", ncols=90, leave=False):
        data_path = leaderboard_root / acc
        forward_dir = recon_root / acc
        forward_dir.mkdir(exist_ok=True)

        if not data_path.exists():
            print(f"Warning: Leaderboard data path not found at {data_path}. Skipping {acc}.")
            continue
            
        # [최종 수정] 리더보드 평가용으로 필요한 모든 인자를 명시적으로 포함하는
        # 새로운 args 객체를 생성하여 모든 오류를 원천 차단합니다.
        recon_args = SimpleNamespace(
            # ===== DataLoader 생성에 필수적인 속성들 =====
            data_path=data_path,
            collator=getattr(args, 'collator', None),
            sampler=getattr(args, 'sampler', None),
            input_key=args.input_key,
            target_key=args.target_key,
            max_key=args.max_key,
            
            # ===== 오류 해결을 위한 명시적 설정 =====
            # 1. CUDA 오류 해결: 자식 프로세스에서 GPU에 접근하지 않도록 num_workers=0으로 강제
            num_workers=0,
            # 2. 배치 크기 오류 해결: isforward=True일 때 사용되는 val_batch_size를 1로 강제
            val_batch_size=1,
            batch_size=1, # 혹시 모를 경우를 대비해 batch_size도 1로 설정

            # ===== test 함수 실행에 필요한 속성 =====
            GPU_NUM=args.GPU_NUM
        )
        
        loader = create_data_loaders(
            data_path=data_path, 
            args=recon_args, 
            isforward=True, 
            classifier=classifier
        )

        reconstructions, _ = test(recon_args, model, loader, classifier)
        save_reconstructions(reconstructions, forward_dir)
    
    # ---------- 2. Evaluation ----------
    ssim = {}
    for acc in tqdm(("acc4", "acc8"), desc="SSIM-eval", ncols=90, leave=False):
        your_data_path = recon_root / acc
        if not any(your_data_path.iterdir()):
             print(f"Warning: No reconstructions found for {acc} at {your_data_path}. Setting SSIM to 0.")
             ssim[acc] = 0.0
             continue

        # [최종 수정] leaderboard_eval.py가 필요로 하는 GPU_NUM을 추가합니다.
        args_eval = SimpleNamespace(
            leaderboard_data_path=leaderboard_root / acc / "image",
            your_data_path=your_data_path,
            output_key=eval_cfg.get("output_key", "reconstruction"),
            GPU_NUM=args.GPU_NUM
        )
        ssim[acc] = eval_mod.forward(args_eval)

    ssim["mean"] = (ssim.get("acc4", 0) + ssim.get("acc8", 0)) / 2.0
    return ssim
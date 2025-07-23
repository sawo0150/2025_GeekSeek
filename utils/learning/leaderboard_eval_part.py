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

# [PROMPT-MR] 필요한 모듈 직접 임포트
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions
# leaderboard_eval.py 모듈은 SSIM 계산을 위해 계속 사용
try:
    eval_mod = importlib.import_module("leaderboard_eval")
except ImportError:
    print("Warning: 'leaderboard_eval' module not found. SSIM calculation will fail.")
    eval_mod = None

# test 함수는 test_part에서 가져오지 않고, 이 파일 내에서 간단히 재정의하거나
# test_part.py에 대한 의존성을 명시적으로 둠. 여기서는 후자를 선택.
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

    eval_cfg = args.evaluation
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
            
        recon_args = SimpleNamespace(
            GPU_NUM=args.GPU_NUM,
            batch_size=eval_cfg.get("batch_size", 1),
            data_path=data_path,
            forward_dir=forward_dir,
            num_workers=args.num_workers,
            input_key=args.input_key,
            collator=getattr(args, 'collator', None),
            sampler=getattr(args, 'sampler', None),
            use_crop=getattr(args, 'use_crop', False),
            centerCropPadding=getattr(args, 'centerCropPadding', None),
            max_key = getattr(args, 'max_key', -1), # isforward=True일 때 필요
            target_key = getattr(args, 'target_key', None) # isforward=True일 때 필요
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

        args_eval = SimpleNamespace(
            leaderboard_data_path=leaderboard_root / acc / "image",
            your_data_path=your_data_path,
            output_key=eval_cfg.get("output_key", "reconstruction"),
        )
        ssim[acc] = eval_mod.forward(args_eval)

    ssim["mean"] = (ssim.get("acc4", 0) + ssim.get("acc8", 0)) / 2.0
    return ssim

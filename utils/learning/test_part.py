# utils/learning/test_part.py
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from hydra.utils import instantiate
from omegaconf import OmegaConf

def test(args, model, data_loader, classifier=None):
    model.eval()
    if classifier:
        classifier.eval()

    reconstructions = defaultdict(dict)
    is_prompt_model = "Prompt" in model.__class__.__name__

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Reconstructing", ncols=90, leave=False):
            # [최종 수정] 9개, 8개, 7개, 6개 모든 케이스를 유연하게 처리
            # isforward=True 모드에서는 9개 또는 6개(non-prompt)가 올 수 있습니다.
            if len(batch) == 9: # promptmr + acc_idx
                mask, kspace, _, _, fnames, slices, _, domain_indices, acc_indices = batch
                domain_indices = domain_indices.cuda(non_blocking=True)
                acc_indices = acc_indices.cuda(non_blocking=True)
            elif len(batch) == 8: # (호환성 유지) acc_idx 없는 promptmr
                mask, kspace, _, _, fnames, slices, _, domain_indices = batch
                domain_indices = domain_indices.cuda(non_blocking=True)
                acc_indices = None
            elif len(batch) == 6: # 일반 forward 모드
                mask, kspace, _, _, fnames, slices = batch
                domain_indices = None
                acc_indices = None
            else:
                 raise ValueError(f"Unexpected data batch length in test function: {len(batch)}")

            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            
            # [최종 수정] 모델 종류에 따라 필요한 인자를 정확히 전달
            if "PromptFIVarNet" in model.__class__.__name__:
                output = model(kspace, mask, domain_indices, acc_indices)
            elif is_prompt_model:
                output = model(kspace, mask, domain_indices)
            else:
                output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args, classifier=None):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    model_cfg = getattr(args, "model", {"_target_": "utils.model.varnet.VarNet"})
    model = instantiate(OmegaConf.create(model_cfg), use_checkpoint=False)
    model.to(device=device)
    
    checkpoint_path = Path(args.exp_dir) / 'best_model.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    state_dict = checkpoint['model']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    print(f"Loaded reconstruction model from epoch {checkpoint.get('epoch', 'N/A')}")
    
    forward_loader = create_data_loaders(
        data_path=args.data_path, 
        args=args, 
        isforward=True, 
        classifier=classifier
    )
    reconstructions, inputs = test(args, model, forward_loader, classifier=classifier)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)

    del forward_loader
    torch.cuda.empty_cache()
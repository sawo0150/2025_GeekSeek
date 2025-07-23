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
            if len(batch) == 8: # isforward=False for val set, returns 8
                mask, kspace, _, _, fnames, slices, _, domain_indices = batch
            elif len(batch) == 7: # isforward=True for test set with prompt model
                mask, kspace, _, _, fnames, slices, domain_indices = batch
            else: # default test set, old model
                mask, kspace, _, _, fnames, slices = batch
            
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            
            if is_prompt_model:
                domain_indices = domain_indices.cuda(non_blocking=True)
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
    
    # [PROMPT-MR] DeepSpeed 등으로 인해 state_dict key가 다를 수 있어 유연하게 로드
    # 예: 'module.sens_net.norm_unet.unet.down_sample_layers.0.0.weight' -> 'sens_net...'
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

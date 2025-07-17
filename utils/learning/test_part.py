import numpy as np
import torch

from tqdm import tqdm  # 추가
from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from hydra.utils import instantiate
from omegaconf import OmegaConf

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    
    with torch.no_grad():
        # 진행률 표시를 위한 tqdm 래퍼
        for batch in tqdm(data_loader, desc=f"Reconstructing", ncols=70, leave=False):
        # for (mask, kspace, _, _, fnames, slices) in data_loader:
        # for batch in data_loader:
            # forward 모드(6-튜플) / train 모드(7-튜플) 모두 지원
            if len(batch) == 7:
                mask, kspace, _, _, fnames, slices, _ = batch
            else:
                mask, kspace, _, _, fnames, slices = batch
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)
            # print(kspace.shape)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    return reconstructions, None


def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    # print ('Current cuda device ', torch.cuda.current_device())

    # model = VarNet(num_cascades=args.cascade, 
    #                chans=args.chans, 
    #                sens_chans=args.sens_chans)

    # 모델 설정은 args.model dict (_target_ + 파라미터) 기준으로 instantiate
    model_cfg = getattr(args, "model", {"_target_": "utils.model.varnet.VarNet"})
    model = instantiate(OmegaConf.create(model_cfg),
                        use_checkpoint=getattr(args, "training_checkpointing", False))
    model.to(device=device)
    
    checkpoint = torch.load(args.exp_dir / 'best_model.pt', map_location='cpu', weights_only=False)
    print("checkpoint_epoch : ", checkpoint['epoch'], "best_val_loss : ", checkpoint['best_val_loss'].item())
    model.load_state_dict(checkpoint['model'])
    
    # print(args.batch_size)
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)

    del forward_loader
    torch.cuda.empty_cache()
# utils/data/collator.py
import copy, torch
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from collections import defaultdict
from torch.utils.data.dataloader import default_collate
import random

# class IdentityCollator:
#     """
#     그냥 default_collate(batch) 를 호출하는 래퍼.
#     Hydra instantiate 로 안전하게 사용할 수 있습니다.
#     """
#     def __call__(self, batch):
#         return torch.utils.data.dataloader.default_collate(batch)
    
class IdentityCollator:
    """
    default_collate(batch) 후,
    debug=True 면 mask, kspace, target 텐서의 shape을 출력합니다.
    """
    def __init__(self, debug: bool = False):
        self.debug = debug

    def __call__(self, batch):
        out = default_collate(batch)

        if self.debug:
            # train_epoch 에서 기대하는 튜플 순서:
            # mask, kspace, target, maximum, fnames, slices, cats = out
            mask, kspace, target, *_ = out
            print(f"[COLLATE DEBUG] mask:  {tuple(mask.shape)}")
            print(f"[COLLATE DEBUG] kspace:{tuple(kspace.shape)}")
            print(f"[COLLATE DEBUG] target:{tuple(target.shape)}")

        return out
class DynamicCompressCollator:
    def __init__(self, compressor: DictConfig):
        # compress_cfg는 SimpleNamespace 내부의 DictConfig 객체일 수 있으므로
        # OmegaConf.create로 감싸주는 것이 안전합니다.
        self.compress_cfg = compressor
        
        # target_coils가 'auto'가 아닐 경우, 미리 고정된 compressor를 생성합니다.
        if self.compress_cfg.target_coils != 'auto':
            self.static_compressor = instantiate(compressor)
        else:
            self.static_compressor = None

    def __call__(self, batch):
        # batch: [(mask, k, tgt, ...), (mask, k, tgt, ...), ...] 형태의 리스트
        
        # 1. 현재 배치에서 사용할 compressor 결정
        if self.static_compressor:
            compressor = self.static_compressor
        else: # 'auto' 모드
            coil_counts = [sample[1].shape[0] for sample in batch]
            target_coils = min(coil_counts)
            
            # print(f"✔️  [Collator] Dynamic compression: Target coils = {target_coils} for this batch.")

            cfg = copy.deepcopy(self.compress_cfg)            
            # current_cfg = self.compress_cfg.copy()
            cfg.target_coils = target_coils
            compressor = instantiate(cfg)

        # 2. 결정된 compressor를 사용하여 배치 내 모든 샘플 처리
        processed_batch = []
        for sample in batch:
            # ✨ BaseCompressor의 __call__을 직접 호출하여 압축 수행
            # sample은 (mask, kspace, target, attrs, fname, slice_idx, cat) 튜플
            processed_sample = compressor(*sample[:-1]) # 마지막 'cat' 제외하고 전달
            # compressor.__call__은 (mask, k_comp, target, ...) 튜플을 반환
            
            # unpack
            mask_p, kspace_p, target_p, attrs_p, fname_p, slice_idx_p = processed_sample

            # 이제 tensor만 shape 출력
            # print(f"mask_p:  {mask_p.shape}")
            # print(f"kspace_p:{kspace_p.shape}")
            # print(f"target_p:{target_p.shape}")
            # attrs_p, fname_p, slice_idx_p 는 텐서가 아닐 수 있으니
            # print(f"attrs_p type: {type(attrs_p)}, fname_p: {fname_p}, slice_idx_p: {slice_idx_p}")

            # 원래 튜플의 마지막 요소였던 category('cat')를 다시 붙여줍니다.
            processed_batch.append((*processed_sample, sample[-1]))

        # 3. 처리된 샘플 리스트를 기본 collate 함수에 넘겨 최종 배치 텐서 생성
        return torch.utils.data.dataloader.default_collate(processed_batch)

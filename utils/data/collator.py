# utils/data/collator.py
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

class DynamicCompressCollator:
    def __init__(self, compress_cfg):
        # compress_cfg는 SimpleNamespace 내부의 DictConfig 객체일 수 있으므로
        # OmegaConf.create로 감싸주는 것이 안전합니다.
        self.compress_cfg = OmegaConf.create(compress_cfg)
        
        # target_coils가 'auto'가 아닐 경우, 미리 고정된 compressor를 생성합니다.
        if self.compress_cfg.target_coils != 'auto':
            self.compressor = instantiate(self.compress_cfg)
        else:
            self.compressor = None

    def __call__(self, batch):
        # batch: [(mask, k, tgt, ...), (mask, k, tgt, ...), ...] 형태의 리스트
        
        # 1. 현재 배치에서 사용할 compressor 결정
        if self.compressor:
            compressor = self.compressor
        else: # 'auto' 모드
            coil_counts = [sample[1].shape[0] for sample in batch]
            target_coils = min(coil_counts)
            
            # print(f"✔️  [Collator] Dynamic compression: Target coils = {target_coils} for this batch.")

            current_cfg = self.compress_cfg.copy()
            current_cfg.target_coils = target_coils
            compressor = instantiate(current_cfg)

        # 2. 결정된 compressor를 사용하여 배치 내 모든 샘플 처리
        processed_batch = []
        for sample in batch:
            # ✨ BaseCompressor의 __call__을 직접 호출하여 압축 수행
            # sample은 (mask, kspace, target, attrs, fname, slice_idx, cat) 튜플
            processed_sample = compressor(*sample[:-1]) # 마지막 'cat' 제외하고 전달
            # compressor.__call__은 (mask, k_comp, target, ...) 튜플을 반환
            
            # 원래 튜플의 마지막 요소였던 category('cat')를 다시 붙여줍니다.
            processed_batch.append((*processed_sample, sample[-1]))

        # 3. 처리된 샘플 리스트를 기본 collate 함수에 넘겨 최종 배치 텐서 생성
        return torch.utils.data.dataloader.default_collate(processed_batch)
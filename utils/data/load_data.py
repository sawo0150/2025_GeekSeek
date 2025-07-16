# utils/data/load_data.py

import h5py, re
import random
from utils.data.transforms import DataTransform, ImageSpaceCropTransform # ✨ ImageSpaceCropTransform 임포트
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np
from torch.utils.data import default_collate
from torch.utils.data.sampler import BatchSampler
from utils.data.transform_wrapper import TransformWrapper
from utils.data.sampler import GroupByCoilBatchSampler

class MultiCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
        # ✨ MRAugmenter가 튜플의 마지막 cat을 처리하지 못하므로, 임시로 분리 후 다시 합칩니다.
        # SliceData.__getitem__의 반환값에 따라 달라집니다.
        
        current_sample = (mask, kspace, target, attrs, fname, slice_idx)
        for tr in self.transforms:
            # print(f"Applying transform: {tr.__class__.__name__}")
            current_sample = tr(*current_sample)
        return current_sample


class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []

        def _cat(fname: Path, attrs: dict): # ✨ attrs에서 카테고리 정보 추출
            # 'CORPD_FBK' 같은 정보가 attrs에 있음
            acq = attrs.get('acquisition', 'unknown')
            
            organ = "brain" if "brain" in str(fname).lower() else "knee"

            # 가속도 정보는 attrs에서 찾는 것을 우선으로 함
            if 'acceleration' in attrs:
                acc_val = attrs['acceleration']
                acc = f"x{acc_val}"
            else: # 없으면 파일명에서 추측
                acc = "x4" if re.search(r"_acc4_|x4|r04", fname.name, re.I) else "x8"
                
            return f"{organ}_{acc}"
        
        image_files = sorted(list(Path(root / "image").iterdir())) if (root / "image").exists() else []
        kspace_files = sorted(list(Path(root / "kspace").iterdir()))

        # 파일 목록 생성
        if not forward:
            file_map = {f.name: f for f in image_files}
            for ksp_fname in kspace_files:
                if ksp_fname.name in file_map:
                    img_fname = file_map[ksp_fname.name]
                    with h5py.File(ksp_fname, 'r') as hf:
                        num_slices = hf[self.input_key].shape[0]
                        attrs = dict(hf.attrs)
                        cat = _cat(ksp_fname, attrs)
                    for i in range(num_slices):
                        self.image_examples.append((img_fname, i, cat))
                        self.kspace_examples.append((ksp_fname, i, cat))
        else: # forward 모드
            for fname in kspace_files:
                with h5py.File(fname, 'r') as hf:
                    num_slices = hf[self.input_key].shape[0]
                    attrs = dict(hf.attrs)
                    cat = _cat(fname, attrs) # forward 시에도 cat 정보 생성
                for i in range(num_slices):
                    self.kspace_examples.append((fname, i, cat))
            self.image_examples = self.kspace_examples

        shape_map = {}
        for entry in self.kspace_examples:
            fname = entry[0]
            key = str(fname)
            if key not in shape_map:
                with h5py.File(fname, 'r') as hf:
                    arr = hf[self.input_key]
                    shape_map[key] = tuple(arr.shape[1:])

        self.coil_counts = [shape[0] for shape in (shape_map[str(e[0])] for e in self.kspace_examples)]
        self.sample_shapes = [shape_map[str(entry[0])] for entry in self.kspace_examples]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            return hf[self.input_key].shape[0]

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        kspace_fname, dataslice, cat = self.kspace_examples[i]
        
        with h5py.File(kspace_fname, "r") as hf:
            input_data = hf[self.input_key][dataslice]
            mask = np.array(hf["mask"])
            attrs = dict(hf.attrs)
            attrs['cat'] = cat # ✨ attrs에 카테고리 정보 명시적으로 추가

        if self.forward:
            target = -1
        else:
            image_fname, _, _ = self.image_examples[i]
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]

        sample_tuple = self.transform(mask, input_data, target, attrs, kspace_fname.name, dataslice)

        # ✨ transform 파이프라인의 출력이 튜플이므로, cat을 마지막에 추가
        return (*sample_tuple, cat)


def create_data_loaders(data_path, args, shuffle=False, isforward=False,
                        augmenter=None, mask_augmenter=None, is_train=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = None

    transforms = []

    # ✨ [수정] 오류가 발생한 부분
    # 1. 코일 압축 (Optional)
    compressor_cfg = getattr(args, "compressor", None)
    # 딕셔너리 키 접근 방식으로 변경하고, .get()으로 안전하게 접근
    if compressor_cfg and compressor_cfg.get('_target_') != "utils.data.coil_compression.IdentityCompressor":
        print(f"✔️ [Transform] Compressor 활성화: {compressor_cfg.get('_target_')}")
        # instantiate는 DictConfig 객체를 기대하므로, dict를 다시 OmegaConf 객체로 변환
        comp_tr = instantiate(OmegaConf.create(compressor_cfg))
        transforms.append(comp_tr)

    # ... (이하 파이프라인은 이전과 동일) ...
    # 2. 이미지 공간 크롭/패딩 (use_crop 플래그로 제어)
    if getattr(args, 'use_crop', False):
        transforms.append(
            ImageSpaceCropTransform(
                target_size=tuple(args.crop_size),
                corner_size=getattr(args, 'corner_size', 16)
            )
        )
    
    # 3. MRAugmenter (이미지 공간 증강)
    if augmenter is not None and not isforward:
        print("✔️ [Transform] MRAugmenter 활성화")
        transforms.append(augmenter)

    # 4. MaskAugmenter (마스크 패턴 증강)
    if mask_augmenter is not None and not isforward:
        print("✔️ [Transform] MaskAugmenter 활성화")
        transforms.append(mask_augmenter)
    
    # 5. 마스크 적용
    from utils.data.transforms import MaskApplyTransform
    transforms.append(MaskApplyTransform())
    
    # 6. 최종 텐서 변환
    transforms.append(DataTransform(isforward, max_key_))

    transform_chain = MultiCompose(transforms)
    
    # ... (이하 Dataset, DataLoader 생성 로직은 이전과 동일) ...
    raw_ds = SliceData(
        root=data_path, transform=lambda *x: x,
        input_key=args.input_key, target_key=target_key_, forward=isforward
    )
    
    dup_cfg = getattr(args,"maskDuplicate",{"enable":False})
    if dup_cfg.get("enable",False) and not isforward and is_train:
        print("✔️ [Dataset] MaskDuplicate 활성화")
        dup_cfg_clean = OmegaConf.create({k:v for k,v in dup_cfg.items() if k!="enable"})
        duped_ds = instantiate(dup_cfg_clean, base_ds=raw_ds, _recursive_=False)
    else:
        duped_ds = raw_ds

    data_storage = TransformWrapper(duped_ds, transform_chain)
    
    collate_fn = default_collate if isforward else instantiate(args.collator, _recursive_=False)
    
    batch_size = args.batch_size if is_train else args.val_batch_size

    if isforward:
        sampler = GroupByCoilBatchSampler(
            data_source=data_storage,
            sample_shapes=getattr(data_storage, "sample_shapes", None),
            batch_size=batch_size,
            shuffle=False,
        )
    else:
        sampler = instantiate(
            args.sampler, data_source=data_storage,
            sample_shapes=getattr(data_storage, "sample_shapes", None),
            batch_size=batch_size,
            shuffle=shuffle,
            _recursive_=False
        )
    
    num_workers = args.num_workers

    if isinstance(sampler, BatchSampler):
        return DataLoader(
            dataset=data_storage,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    else:
        return DataLoader(
            dataset=data_storage,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
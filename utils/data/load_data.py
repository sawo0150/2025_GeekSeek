import h5py, re
import random
from utils.data.transforms import DataTransform, CenterCropOrPad
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf   # ← 추가
from pathlib import Path
import numpy as np
from torch.utils.data import default_collate
from torch.utils.data.sampler import BatchSampler
from utils.data.transform_wrapper import TransformWrapper
# ─ leaderboard forward 전용 기본 샘플러
from utils.data.sampler import GroupByCoilBatchSampler

# +++ custom multi-input Compose +++
class MultiCompose:
    """
    mask, kspace, target, attrs, fname, slice 6-tuple을
    self.transforms 리스트에 있는 transform들에
    차례대로 넘겨주고 최종 결과를 리턴합니다..
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, mask, kspace, target, attrs, fname, slice_idx):
        for tr in self.transforms:
            mask, kspace, target, attrs, fname, slice_idx = tr(
                mask, kspace, target, attrs, fname, slice_idx
            )
        return mask, kspace, target, attrs, fname, slice_idx

class SliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward          # test/submit 모드
        self.image_examples = []        # [(fname, slice, cat)]
        self.kspace_examples = []       # [(fname, slice, cat)], forward=True 일 때는 cat 없음

        # ❶ organ/acc 추출 함수 ---------------------------------------------
        def _cat(fname: Path):
            organ = "brain" if "brain" in fname.name.lower() else "knee"
            acc   = "x4"   if re.search(r"_acc4_|x4|r04", fname.name, re.I) else "x8"
            return f"{organ}_{acc}"
        
        image_files = list(Path(root / "image").iterdir())
        kspace_files = list(Path(root / "kspace").iterdir())
        if not forward:
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)
                cat = _cat(fname)
                self.image_examples += [
                    (fname, slice_ind, cat) for slice_ind in range(num_slices)
                ]

            for fname in sorted(kspace_files):
                num_slices = self._get_metadata(fname)
                cat = _cat(fname)
                self.kspace_examples += [
                    (fname, slice_ind, cat) for slice_ind in range(num_slices)
                ]

        else:       # ★ forward 모드 : cat 생성 금지 + image_label X
            # for fname in sorted(image_files):
            #     num_slices = self._get_metadata(fname)
            #     self.image_examples += [
            #         (fname, slice_ind) for slice_ind in range(num_slices)
            #     ]

            for fname in sorted(kspace_files):
                num_slices = self._get_metadata(fname)
                self.kspace_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices)
                ]
            
            # image_examples 은 dummy 로 동일 길이 맞춰 주기
            self.image_examples = self.kspace_examples

        # --- coil_counts 미리 계산 ---
        # 파일 단위로 한 번씩만 열어서, self.input_key의 두 번째 차원이 C(coils)인 걸 이용
        coil_map = {}
        for entry in self.kspace_examples:
            fname = entry[0]              # 첫 원소만 사용
            key = str(fname)
            if key not in coil_map:
                with h5py.File(fname, 'r') as hf:
                    arr = hf[self.input_key]
                    # arr.shape == (num_slices, C, H, W) 라고 가정
                    coil_map[key] = arr.shape[1]
        # 인덱스 순서대로 coil count 리스트
        self.coil_counts = [coil_map[str(entry[0])] for entry in self.kspace_examples]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            # (1) k-space 파일에는 self.input_key 가 반드시 있음
            if self.input_key in hf:
                return hf[self.input_key].shape[0]

            # (2) label 있는 경우에만 (str 인지 먼저 확인)
            if isinstance(self.target_key, str) and self.target_key in hf:
                return hf[self.target_key].shape[0]

            # (3) 마지막 안전장치 : 첫 데이터셋 사용
            first_ds = next(iter(hf.keys()))
        return hf[first_ds].shape[0]

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _, cat = self.image_examples[i]
            kspace_fname, dataslice, cat = self.kspace_examples[i]
        else:   # forward 모드: cat 없음
            image_fname, _ = self.image_examples[i]
            kspace_fname, dataslice = self.kspace_examples[i]

        if not self.forward and image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = {}      # Mapping 보장

        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        sample = self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)
        return (*sample, cat) if not self.forward else sample


def create_data_loaders(data_path, args, shuffle=False, isforward=False, augmenter=None):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = None

    transforms = []

    # (0) MRaugment
    if augmenter is not None and not isforward:
        transforms.append(augmenter)

    aug_cfg = getattr(args, "maskAugment", {"enable": False})
    if aug_cfg.get("enable", False) and not isforward:
        transforms.append(instantiate(aug_cfg))


    # (1) Mask 적용 및 k-space numpy 반환
    from utils.data.transforms import MaskApplyTransform
    transforms.append(MaskApplyTransform())

    # (2) Spatial crop (토글)
    if getattr(args, 'use_crop', False):
        transforms.append(CenterCropOrPad(target_size=tuple(args.crop_size)))

    # (2) Spatial crop (토글)
    if getattr(args, 'use_crop', False):
        transforms.append(CenterCropOrPad(target_size=tuple(args.crop_size)))

    # # (3) Coil compression (토글)
    # if getattr(args, "compressor", None):
    #     comp_tr = instantiate(args.compressor)
    #     transforms.append(comp_tr)

    # (4) Tensor 변환 및 real/imag 스택
    transforms.append(DataTransform(isforward, max_key_))

    transform_chain = MultiCompose(transforms)

    # 1)  *** Raw SliceData (transform=None) ***
    raw_ds = SliceData(
        root=data_path,
        transform=lambda *x: x,   # identity
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    # 2)  *** Duplicate 적용 (crop 前) ***
    dup_cfg = getattr(args,"maskDuplicate",{"enable":False})
    if dup_cfg.get("enable",False) and not isforward:
        dup_cfg_clean = OmegaConf.create({k:v for k,v in dup_cfg.items()
                                          if k!="enable"})
        duped_ds = instantiate(dup_cfg_clean, base_ds=raw_ds,
                               _recursive_=False)
    else:
        duped_ds = raw_ds

    # 3)  *** TransformWrapper 로 실제 변환 ***
    data_storage = TransformWrapper(duped_ds, transform_chain)
    
    # 1) collate_fn
    if isforward:                       # forward 모드 : 안전 collator
        collate_fn = default_collate
    else:
        collate_fn = instantiate(args.collator, _recursive_=False)

    # 2) sampler 인스턴스 하나만 만들기
    if isforward:
        # ⭐ 리더보드/submit 모드 : 무조건 GroupByCoilBatchSampler 사용
        sampler = GroupByCoilBatchSampler(
            data_source=data_storage,
            coil_counts=getattr(data_storage, "coil_counts", None),
            batch_size=args.batch_size,
            shuffle=False,
        )
    else:
        sampler = instantiate(
            args.sampler,
            data_source=data_storage,
            coil_counts=getattr(data_storage, "coil_counts", None),
            batch_size=args.batch_size,
            shuffle=shuffle,
            _recursive_=False
        )

    # 3) DataLoader 에 넘겨줄 인자 결정
    #    sampler 가 BatchSampler 계열이면 batch_sampler=, 아니면 sampler= 로
    if isinstance(sampler, BatchSampler):
        return DataLoader(
            dataset=data_storage,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )
    else:
        return DataLoader(
            dataset=data_storage,
            sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn
        )

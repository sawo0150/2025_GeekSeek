import h5py, re
import random
from utils.data.transforms import DataTransform, CenterCropOrPad
from torch.utils.data import Dataset, DataLoader
from hydra.utils import instantiate
from pathlib import Path
import numpy as np

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
        self.kspace_examples = []       # [(fname, slice, cat)]

        # ❶ organ/acc 추출 함수 ---------------------------------------------
        def _cat(fname: Path):
            organ = "brain" if "brain" in fname.name.lower() else "knee"
            acc   = "x4"   if re.search(r"_acc4_|x4|r04", fname.name, re.I) else "x8"
            return f"{organ}_{acc}"
        
        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)
                cat = _cat(fname)
                self.image_examples += [
                    (fname, slice_ind, cat) for slice_ind in range(num_slices)
                ]

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)
            cat = _cat(fname)
            self.kspace_examples += [
                (fname, slice_ind, cat) for slice_ind in range(num_slices)
            ]


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _, cat = self.image_examples[i]
        kspace_fname, dataslice, cat = self.kspace_examples[i]
        if not self.forward and image_fname.name != kspace_fname.name:
            raise ValueError(f"Image file {image_fname.name} does not match kspace file {kspace_fname.name}")

        with h5py.File(kspace_fname, "r") as hf:
            input = hf[self.input_key][dataslice]
            mask =  np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)
            
        sample = self.transform(mask, input, target, attrs, kspace_fname.name, dataslice)
        return (*sample, cat)


def create_data_loaders(data_path, args, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1

    transforms = []

    # (0) Dynamic augmentation (추후 통합 예정)
    # # aug_tr = instantiate(args.aug)
    #     # aug_tr,   # TODO: MRaugment 통합 시 여기에 추가

    # (1) Mask 적용 및 k-space numpy 반환
    from utils.data.transforms import MaskApplyTransform
    transforms.append(MaskApplyTransform())

    # (2) Spatial crop (토글)
    if getattr(args, 'use_crop', False):
        transforms.append(CenterCropOrPad(target_size=tuple(args.crop_size)))

    # (3) Coil compression (토글)
    comp_tr = instantiate(args.compress)
    transforms.append(comp_tr)

    # (4) Tensor 변환 및 real/imag 스택
    transforms.append(DataTransform(isforward, max_key_))

    transform = MultiCompose(transforms)


    data_storage = SliceData(
        root=data_path,
        transform=transform,
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers
    )
    return data_loader

import os
from typing import Any, Tuple

import albumentations as A
import numpy as np
import pytorch_lightning as L
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Subset
from torchvision.datasets import Cityscapes as BaseDataset
from torchvision.transforms import v2

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")


class Cityscapes(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment_indices = {}
        self.augments = None

    def set_indices(self, train_indices: list[int], val_indices: list[int]) -> None:
        for index in train_indices:
            self.augment_indices[index] = True

        for index in val_indices:
            self.augment_indices[index] = False

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, raw_mask = super().__getitem__(index)

        raw_mask = np.array(raw_mask)
        mask = np.zeros_like(raw_mask)
        for k in mapping_20:
            target = mapping_20[k]
            if target == 0:
                continue
            mask[raw_mask == k] = target

        if self.augment_indices[index]:
            augmented = self.augments(
                image=(img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8), mask=mask
            )
            img = augmented["image"]
            mask = augmented["mask"]

        if self.transform is not None:
            img = self.transform(img)

        return img, mask, index


PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")

mapping_20 = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 1,
    8: 2,
    9: 0,
    10: 0,
    11: 3,
    12: 4,
    13: 5,
    14: 0,
    15: 0,
    16: 0,
    17: 6,
    18: 0,
    19: 7,
    20: 8,
    21: 9,
    22: 10,
    23: 11,
    24: 12,
    25: 13,
    26: 14,
    27: 15,
    28: 16,
    29: 0,
    30: 0,
    31: 17,
    32: 18,
    33: 19,
    -1: 0,
}


class CityscapesDataModule(L.LightningDataModule):
    classes = [
        "background",
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]

    augments = None

    task = "segmentation"

    def __init__(
        self,
        augmentation_policy_path,
        batch_size: int = 6,
        train_mode: str = "fine",
        data_dir: str = PATH_DATASETS,
    ):
        super().__init__()

        assert os.path.exists(augmentation_policy_path)
        self.augments = A.load(augmentation_policy_path, data_format="yaml")
        self.data_dir = os.path.join(data_dir, "Cityscapes")
        self.num_classes = len(self.classes)
        self.batch_size = batch_size
        self.train_mode = train_mode

        self.transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )

    def remove_item(self, index: int) -> None:
        del self.data[index]
        del self.targets[index]

    def prepare_data(self):
        assert os.path.exists(self.data_dir)

    def remove_train_example(self, idx):
        del self.cityscapes_train.indices[idx]

    def reset_train_data(self):
        self.cityscapes_train.indices = self.initial_train_indices.copy()

    def remove_train_data(self, indices_to_remove):
        self.cityscapes_train.indices = list(
            set(self.cityscapes_train.indices) - set(indices_to_remove)
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cityscapes_fine = Cityscapes(
                self.data_dir,
                split="train",
                mode="fine",
                target_type="semantic",
                transform=self.transform,
            )

            cityscapes_coarse = Cityscapes(
                self.data_dir,
                split="train",
                mode="coarse",
                target_type="semantic",
                transform=self.transform,
            )
            train_size = int(len(cityscapes_fine) * 0.8)
            test_size = int(len(cityscapes_fine) - train_size)

            self.cityscapes_train, self.cityscapes_val = random_split(
                cityscapes_fine, [train_size, test_size]
            )

            if self.train_mode == "coarse":
                self.cityscapes_train = Subset(
                    cityscapes_coarse, self.cityscapes_train.indices
                )

            cityscapes_coarse.set_indices(
                self.cityscapes_train.indices, self.cityscapes_val.indices
            )
            cityscapes_fine.set_indices(
                self.cityscapes_train.indices, self.cityscapes_val.indices
            )
            cityscapes_coarse.augments = self.augments
            cityscapes_fine.augments = self.augments

        if stage == "test" or stage is None:
            self.cityscapes_test = Cityscapes(
                self.data_dir,
                split="val",
                mode="fine",
                target_type="semantic",
                transform=self.transform,
            )
            self.cityscapes_test.set_indices([], range(len(self.cityscapes_test)))

    def train_dataloader(self):
        return DataLoader(
            self.cityscapes_train,
            num_workers=os.cpu_count(),
            shuffle=True,
            batch_size=self.batch_size,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cityscapes_val,
            num_workers=os.cpu_count(),
            shuffle=False,
            batch_size=1,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cityscapes_test,
            num_workers=os.cpu_count(),
            shuffle=False,
            batch_size=1,
            persistent_workers=True,
        )


__all__ = ["CityscapesDataModule"]

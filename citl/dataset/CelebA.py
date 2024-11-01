import os
from typing import Any, Tuple

import albumentations as A
import numpy as np
import pytorch_lightning as L
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA as BaseDataset
from torchvision.transforms import v2

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")


class CelebA(BaseDataset):

    def __init__(self, *args, **kwargs):
        self.resize = kwargs.pop("resize")
        super().__init__(*args, **kwargs)
        self.target_idx = None
        self.sensitive_idx = None
        self.augment_indices = {}
        self.augments = None
        self.target_idx = self.attr_names.index("Wavy_Hair")
        self.sensitive_idx = self.attr_names.index("Male")

    def set_indices(self, train_indices: list[int], val_indices: list[int]) -> None:
        for index in train_indices:
            self.augment_indices[index] = True

        for index in val_indices:
            self.augment_indices[index] = False

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)
        img = self.resize(img)

        train_target = target[self.target_idx]
        sensitive = target[self.sensitive_idx]

        if self.augment_indices[index]:
            augmented = self.augments(image=np.array(img))
            img = augmented["image"]

        img = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(img)

        TRAIN_TARGET_NOT_WAVY = 0
        TRAIN_TARGET_WAVY = 1
        SENSITIVE_FEMALE = 0
        SENSITIVE_MALE = 1

        classes = [
            "Men - Not Wavy",
            "Men - Wavy",
            "Women - Not Wavy",
            "Women - Wavy",
        ]
        if train_target == TRAIN_TARGET_NOT_WAVY and sensitive == SENSITIVE_MALE:
            combined_target = classes.index("Men - Not Wavy")
        if train_target == TRAIN_TARGET_NOT_WAVY and sensitive == SENSITIVE_FEMALE:
            combined_target = classes.index("Women - Not Wavy")
        if train_target == TRAIN_TARGET_WAVY and sensitive == SENSITIVE_MALE:
            combined_target = classes.index("Men - Wavy")
        elif train_target == TRAIN_TARGET_WAVY and sensitive == SENSITIVE_FEMALE:
            combined_target = classes.index("Women - Wavy")

        return img, combined_target, sensitive


PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")


class CelebADataModule(L.LightningDataModule):

    classes = [
        "Men - Not Wavy",
        "Men - Wavy",
        "Women - Not Wavy",
        "Women - Wavy",
    ]

    task = "classification"

    def __init__(
        self,
        augmentation_policy_path,
        batch_size=128,
        data_dir=PATH_DATASETS,
        image_size=176,
    ):
        super().__init__()
        assert os.path.exists(augmentation_policy_path)
        self.augments = A.load(augmentation_policy_path, data_format="yaml")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.image_size = image_size
        self.num_classes = 4

        self.resize = v2.Compose(
            [
                v2.Resize(self.image_size, max_size=self.image_size + 1),
                v2.CenterCrop(self.image_size),
            ]
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.celeba_train = CelebA(
                self.data_dir,
                split="train",
                resize=self.resize,
                target_type="attr",
            )
            self.celeba_val = CelebA(
                self.data_dir,
                split="valid",
                resize=self.resize,
                target_type="attr",
            )
            self.celeba_train.set_indices(range(len(self.celeba_train)), [])
            self.celeba_val.set_indices([], range(len(self.celeba_val)))
            self.celeba_train.augments = self.augments

        if stage == "test":
            self.celeba_test = CelebA(
                self.data_dir, split="test", resize=self.resize, target_type="attr"
            )
            self.celeba_test.set_indices([], range(len(self.celeba_test)))

    def train_dataloader(self):
        return DataLoader(
            self.celeba_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.celeba_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.celeba_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            persistent_workers=True,
        )

import os
import zipfile
from functools import partial
from os.path import join
from typing import Any, Tuple

import albumentations as A
import numpy as np
import pandas as pd
import PIL
import pytorch_lightning as L
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.utils import (
    check_integrity,
    download_file_from_google_drive,
    verify_str_arg,
)
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import v2
from torchvision.datasets import CelebA as BaseDataset

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")

class CelebA(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.augment_indices = {}
        self.augments = None
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.target_idx = None
        self.sensitive_idx = None

    def set_indices(self, train_indices: list[int], val_indices: list[int]) -> None:
        for index in train_indices:
            self.augment_indices[index] = True

        for index in val_indices:
            self.augment_indices[index] = False

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)

        if self.target_idx is None:
            self.target_idx = self.attr_names.index("Wavy_Hair")
        if self.sensitive_idx is None:
            self.sensitive_idx = self.attr_names.index("Male")

        train_target = target[self.target_idx]
        sensitive = target[self.sensitive_idx]

        if self.transform is not None:
            img = self.transform(img)

        if self.augment_indices[index]:
            augmented = self.augments(image=img.numpy().transpose(1, 2, 0))
            img = augmented["image"]

        img = self.normalize(img)

        return img, train_target, sensitive


PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")


class CelebADataModule(L.LightningDataModule):

    classes = [
        "Not Wavy",
        "Wavy",
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
        self.num_classes = 2

        self.transform = v2.Compose(
            [
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Resize(self.image_size, max_size=self.image_size + 1),
                v2.CenterCrop(self.image_size),
                v2.ToTensor(),
            ]
        )

    def prepare_data(self):
        CelebA(self.data_dir, split="train")
        CelebA(self.data_dir, split="valid")
        CelebA(self.data_dir, split="test")

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.celeba_train = CelebA(
                self.data_dir, split="train", transform=self.transform, target_type="attr"
            )
            self.celeba_val = CelebA(
                self.data_dir, split="valid", transform=self.transform, target_type="attr"
            )
            self.celeba_train.set_indices(range(len(self.celeba_train)), [])
            self.celeba_val.set_indices([], range(len(self.celeba_val)))
            self.celeba_train.augments = self.augments

        if stage == "test" or stage is None:
            self.celeba_test = CelebA(
                self.data_dir, split="test", transform=self.transform, target_type="attr"
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

import os
from typing import Any, Tuple

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
import pytorch_lightning as L
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")
EXPLICIT_BIAS = False
TARGET_IDX = "Wavy_Hair"
SENSITIVE_IDX = "Male"

class CelebA(Dataset):

    def __init__(self, data_dir, split="train"):
        self.data_dir = os.path.join(data_dir, "celeba", "processed", split)
        self.attrs = pd.read_csv(f"{self.data_dir}/_attrs.csv")
        self.attr_names = self.attrs.columns.tolist()
        self.split = split
        self.augment_indices = {}
        self.augments = None
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.target_idx = self.attr_names.index(TARGET_IDX)
        self.sensitive_idx = self.attr_names.index(SENSITIVE_IDX)

        self.images = []

    def set_indices(self, train_indices: list[int], val_indices: list[int]) -> None:
        for index in train_indices:
            self.augment_indices[index] = True

        for index in val_indices:
            self.augment_indices[index] = False

    def __len__(self):
        return len(self.attrs)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(f"{self.data_dir}/{index}.png")
        target = self.attrs.iloc[index]

        train_target = target.iloc[self.target_idx]
        sensitive = target.iloc[self.sensitive_idx]

        if self.augment_indices[index]:
            augmented = self.augments(image=np.array(img))
            img = augmented["image"]

        img = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), self.normalize]
        )(img)

        if EXPLICIT_BIAS:
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
                train_target = classes.index("Men - Not Wavy")
            if train_target == TRAIN_TARGET_NOT_WAVY and sensitive == SENSITIVE_FEMALE:
                train_target = classes.index("Women - Not Wavy")
            if train_target == TRAIN_TARGET_WAVY and sensitive == SENSITIVE_MALE:
                train_target = classes.index("Men - Wavy")
            elif train_target == TRAIN_TARGET_WAVY and sensitive == SENSITIVE_FEMALE:
                train_target = classes.index("Women - Wavy")

        return img, train_target, sensitive


PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")


class CelebADataModule(L.LightningDataModule):

    classes = (
        [
            "Not Wavy",
            "Wavy",
        ]
        if not EXPLICIT_BIAS
        else [
            "Men - Not Wavy",
            "Men - Wavy",
            "Women - Not Wavy",
            "Women - Wavy",
        ]
    )

    task = "classification"

    def __init__(
        self,
        augmentation_policy_path,
        batch_size=128,
        data_dir=PATH_DATASETS,
    ):
        super().__init__()
        assert os.path.exists(augmentation_policy_path)
        self.augments = A.load(augmentation_policy_path, data_format="yaml")
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_classes = 2

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.celeba_train = CelebA(
                self.data_dir,
                split="train",
            )
            self.celeba_val = CelebA(
                self.data_dir,
                split="valid",
            )
            self.celeba_train.set_indices(range(len(self.celeba_train)), [])
            self.celeba_val.set_indices([], range(len(self.celeba_val)))
            self.celeba_train.augments = self.augments

        if stage == "test":
            self.celeba_test = CelebA(self.data_dir, split="test")
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

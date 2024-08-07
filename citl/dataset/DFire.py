import os
from typing import Any, Tuple

import albumentations as A
import numpy as np
import pytorch_lightning as L
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2

from .YoloDataset import YoloDataset

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./datasets")


def remove_extensions(file_list):
    return [os.path.splitext(filename)[0] for filename in file_list]


class DFire(YoloDataset):
    def __init__(self, data_dir: str, train: bool, transform=None):
        self.augment_indices = {}
        self.augments = None
        self.transform = transform

        if train:
            self.data_dir = os.path.join(data_dir, "DFire", "processed", "train")
        else:
            self.data_dir = os.path.join(data_dir, "DFire", "processed", "test")

        images_dir = os.path.join(self.data_dir, "images")
        labels_dir = os.path.join(self.data_dir, "labels")
        super().__init__(images_dir, labels_dir)

    def set_indices(self, train_indices: list[int], val_indices: list[int]) -> None:
        for index in train_indices:
            self.augment_indices[index] = True

        for index in val_indices:
            self.augment_indices[index] = False

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)

        if self.transform is not None:
            img = self.transform(img)

        if self.augment_indices[index]:
            augmented = self.augments(image=img.numpy().transpose(1, 2, 0))
            img = augmented["image"]

        target = 0 if len(target["labels"]) == 0 else 1

        return img, target, index


class DFireDataModule(L.LightningDataModule):
    classes = ["no fire", "fire"]

    augments = None

    task = "classification"

    def __init__(
        self, augmentation_policy_path, batch_size=128, data_dir: str = PATH_DATASETS
    ):
        super().__init__()

        assert os.path.exists(augmentation_policy_path)
        self.augments = A.load(augmentation_policy_path, data_format="yaml")
        self.data_dir = data_dir
        self.num_classes = 2
        self.batch_size = batch_size

    def set_image_size(self, image_size: int, greyscale: bool):
        self.image_size = image_size
        if greyscale:
            raise NotImplementedError("Greyscale not implemented.")

        self.transform = v2.Compose(
            [v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])]
        )

    def prepare_data(self):
        DFire(self.data_dir, train=True)
        DFire(self.data_dir, train=False)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dfire_full = DFire(self.data_dir, train=True, transform=self.transform)
            num_samples = len(dfire_full)
            num_val_samples = num_samples // 10
            num_train_samples = num_samples - num_val_samples
            self.dfire_train, self.dfire_val = random_split(
                dfire_full, [num_train_samples, num_val_samples]
            )
            dfire_full.set_indices(self.dfire_train.indices, self.dfire_val.indices)
            dfire_full.augments = self.augments

        if stage == "test" or stage is None:
            self.dfire_test = DFire(
                self.data_dir, train=False, transform=self.transform
            )
            self.dfire_test.set_indices([], range(len(self.dfire_test)))

    def train_dataloader(self):
        return DataLoader(
            self.dfire_train,
            num_workers=4,
            shuffle=True,
            batch_size=self.batch_size,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dfire_val,
            num_workers=4,
            shuffle=False,
            batch_size=self.batch_size,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dfire_test,
            num_workers=4,
            shuffle=False,
            batch_size=self.batch_size,
            persistent_workers=True,
        )


__all__ = ["DFireDataModule"]

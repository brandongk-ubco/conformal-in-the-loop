import os
from typing import Any, Tuple

import albumentations as A
import torch
from PIL import Image
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10

from .CIFAR10 import CIFAR10DataModule


class AugmentedCIFAR10(CIFAR10):
    augment_indices = {}
    augments = None

    def set_indices(self, train_indices: list[int], val_indices: list[int]) -> None:
        for index in train_indices:
            self.augment_indices[index] = True

        for index in val_indices:
            self.augment_indices[index] = False

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        if self.augment_indices[index]:
            augmented = self.augments(image=img)
            img = augmented["image"]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        assert img.shape == (3, 32, 32)

        return img, target


PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")


class AugmentedCIFAR10DataModule(CIFAR10DataModule):
    augments = None

    def __init__(self, augmentation_policy_path, data_dir: str = PATH_DATASETS):
        super().__init__(data_dir=data_dir)

        assert os.path.exists(augmentation_policy_path)

        self.augments = A.load(augmentation_policy_path, data_format="yaml")

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = AugmentedCIFAR10(
                self.data_dir, train=True, transform=self.transform
            )
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])
            cifar_full.set_indices(self.cifar_train.indices, self.cifar_val.indices)
            cifar_full.augments = self.augments

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

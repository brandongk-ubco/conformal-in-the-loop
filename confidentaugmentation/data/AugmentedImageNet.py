import os
from typing import Any, Tuple

import albumentations as A
from torch.utils.data import random_split
from torchvision.datasets import ImageNet
from torchvision.transforms import functional as F

from .ImageNet import ImageNetDataModule


class AugmentedImageNet(ImageNet):
    augment_indices = {}
    augments = None

    def set_indices(self, train_indices: list[int], val_indices: list[int]) -> None:
        for index in train_indices:
            self.augment_indices[index] = True

        for index in val_indices:
            self.augment_indices[index] = False

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.augment_indices[index]:
            augmented = self.augments(image=img.numpy().transpose(1, 2, 0))
            img = augmented["image"]

        F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)

        return img, target, index


PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")


class AugmentedImageNetDataModule(ImageNetDataModule):
    augments = None

    def __init__(self, augmentation_policy_path, data_dir: str = PATH_DATASETS):
        super().__init__(data_dir=data_dir)

        assert os.path.exists(augmentation_policy_path)

        self.augments = A.load(augmentation_policy_path, data_format="yaml")

    def setup(self, stage=None):
        imagenet_dir = os.path.join(self.data_dir, "imagenet")
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            imagenet_full = AugmentedImageNet(
                imagenet_dir, split="train", transform=self.transform
            )
            dataset_size = len(imagenet_full)
            val_size = int(dataset_size * 0.1)
            train_size = dataset_size - val_size
            self.imagenet_train, self.imagenet_val = random_split(
                imagenet_full, [train_size, val_size]
            )
            imagenet_full.set_indices(
                self.imagenet_train.indices, self.imagenet_val.indices
            )
            imagenet_full.augments = self.augments

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.imagenet_test = ImageNet(
                imagenet_dir, split="val", transform=self.transform
            )

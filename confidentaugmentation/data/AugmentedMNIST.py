import os
from typing import Any, Tuple

import albumentations as A
from PIL import Image
from torch.utils.data import random_split

from .MNist import MNISTDataModule
from torchvision.datasets import MNIST



class AugmentedMNIST(MNIST):
    augment_indices = {}
    augments = None
    augmentation_probability = 0.0

    def set_indices(self, train_indices: list[int], val_indices: list[int]) -> None:
        for index in train_indices:
            self.augment_indices[index] = True

        for index in val_indices:
            self.augment_indices[index] = False

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        img = img.numpy()

        if self.augment_indices[index]:
            augmented = self.augments(image=img)
            img = augmented["image"]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        assert img.shape == (1, 32, 32)

        return img, target


PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")


class AugmentedMNISTDataModule(MNISTDataModule):
    augments = None

    def __init__(self, augmentation_policy_path, data_dir: str = PATH_DATASETS):
        super().__init__(data_dir=data_dir)

        assert os.path.exists(augmentation_policy_path)

        self.augments = A.load(augmentation_policy_path, data_format="yaml")

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = AugmentedMNIST(
                self.data_dir, train=True, transform=self.transform, download=True
            )
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
            mnist_full.set_indices(self.mnist_train.indices, self.mnist_val.indices)
            mnist_full.augments = self.augments

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform, download=True
            )

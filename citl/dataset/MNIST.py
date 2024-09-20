import os
from typing import Any, Tuple

import albumentations as A
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST as BaseDataset
from torchvision.transforms import v2

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")


class MNIST(BaseDataset):
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
        img, target = self.data[index], self.targets[index]

        img = img.unsqueeze(0)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.augment_indices[index]:
            augmented = self.augments(image=img.numpy().transpose(1, 2, 0))
            img = augmented["image"]

        img = img.float()
        img = img / img.max()

        return img, target, index


PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")


class MNISTDataModule(L.LightningDataModule):
    classes = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ]

    augments = None

    task = "classification"

    def __init__(
        self,
        augmentation_policy_path,
        batch_size: int = 256,
        data_dir: str = PATH_DATASETS,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.num_classes = 10
        self.batch_size = batch_size

        assert os.path.exists(augmentation_policy_path)

        self.augments = A.load(augmentation_policy_path, data_format="yaml")
        self.image_size = 28
        self.transform = v2.Compose(
            [
                v2.Compose(
                    [
                        v2.ToImage(),
                        v2.ToDtype(torch.uint8, scale=True),
                        v2.ToDtype(torch.float16, scale=True),
                    ]
                ),
                v2.Resize(
                    self.image_size, max_size=self.image_size + 1, antialias=False
                ),
                v2.CenterCrop(self.image_size),
            ]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(
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
            self.mnist_test.set_indices([], range(len(self.mnist_test)))

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            num_workers=os.cpu_count(),
            shuffle=True,
            batch_size=self.batch_size,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            num_workers=os.cpu_count(),
            shuffle=False,
            batch_size=self.batch_size,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            num_workers=os.cpu_count(),
            shuffle=False,
            pin_memory=True,
            batch_size=self.batch_size,
            persistent_workers=True,
        )


__all__ = ["MNISTDataModule"]

import os

import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, random_split
# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")
BATCH_SIZE = 128


class CIFAR10DataModule(L.LightningDataModule):
    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    def __init__(self, data_dir: str = PATH_DATASETS, image_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.transform = v2.Compose(
            [
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Resize(image_size, max_size=image_size + 1, antialias=False),
                v2.CenterCrop(image_size),
            ]
        )

        self.num_classes = 10

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True)
        CIFAR10(self.data_dir, train=False)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            num_workers=os.cpu_count(),
            shuffle=True,
            batch_size=BATCH_SIZE,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            num_workers=os.cpu_count(),
            shuffle=False,
            batch_size=BATCH_SIZE,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            num_workers=os.cpu_count(),
            shuffle=False,
            batch_size=BATCH_SIZE,
            persistent_workers=True,
        )

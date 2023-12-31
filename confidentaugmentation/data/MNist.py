import os

import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST
from torchvision.transforms import v2

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256


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

    def __init__(self, data_dir: str = PATH_DATASETS, image_size=28):
        super().__init__()
        self.data_dir = data_dir
        self.transform = v2.Compose(
            [
                v2.ToTensor(),
                v2.Resize(image_size, max_size=image_size + 1, antialias=False),
                v2.CenterCrop(image_size),
            ]
        )

        self.dims = (1, 28, 28)
        self.num_classes = 10

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            num_workers=os.cpu_count(),
            shuffle=True,
            pin_memory=True,
            batch_size=BATCH_SIZE,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val,
            num_workers=os.cpu_count(),
            shuffle=False,
            pin_memory=True,
            batch_size=BATCH_SIZE,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test,
            num_workers=os.cpu_count(),
            shuffle=False,
            pin_memory=True,
            batch_size=BATCH_SIZE,
            persistent_workers=True,
        )

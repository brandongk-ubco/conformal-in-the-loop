import os

import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import v2
import os
from torchvision.datasets import ImageNet
import json

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")
BATCH_SIZE = 128
CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))


class ImageNetDataModule(L.LightningDataModule):

    def __init__(self, data_dir: str = PATH_DATASETS, image_size=224):
        super().__init__()
        self.data_dir = data_dir
        self.transform = v2.Compose(
            [
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Resize(image_size, max_size=image_size + 1, antialias=False),
                v2.CenterCrop(image_size),
            ]
        )

        with open(os.path.join(CURRENT_DIRECTORY, "imagenet.json"), "r") as f:
            parsed_json = json.load(f)
            self.classes = list(parsed_json.values())
        self.num_classes = len(self.classes)

    def setup(self, stage=None):
        imagenet_dir = os.path.join(self.data_dir, "imagenet")
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            imagenet_full = ImageNet(imagenet_dir, split="train", transform=self.transform)
            dataset_size = len(imagenet_full)
            val_size = int(dataset_size * 0.1)
            train_size = dataset_size - val_size
            self.imagenet_train, self.imagenet_val = random_split(imagenet_full, [train_size, val_size])
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.imagenet_test = ImageNet(
                imagenet_dir, split="val", transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.imagenet_train,
            num_workers=os.cpu_count(),
            shuffle=True,
            batch_size=BATCH_SIZE,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.imagenet_val,
            num_workers=os.cpu_count(),
            shuffle=False,
            batch_size=BATCH_SIZE,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.imagenet_test,
            num_workers=os.cpu_count(),
            shuffle=False,
            batch_size=BATCH_SIZE,
            persistent_workers=True,
        )

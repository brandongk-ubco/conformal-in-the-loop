import os
from typing import Any, Tuple

import albumentations as A
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10 as BaseDataset
from torchvision.transforms import v2

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")
BATCH_SIZE = 128


class CIFAR10(BaseDataset):
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

        if self.transform is not None:
            img = self.transform(img)

        if self.augment_indices[index]:
            augmented = self.augments(image=img.numpy().transpose(1, 2, 0))
            img = augmented["image"]

        return img, target, index


PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")


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

    augments = None

    def __init__(self, augmentation_policy_path, data_dir: str = PATH_DATASETS):
        super().__init__()

        assert os.path.exists(augmentation_policy_path)
        self.augments = A.load(augmentation_policy_path, data_format="yaml")
        self.data_dir = data_dir
        self.num_classes = 10

    def remove_item(self, index: int) -> None:
        del self.data[index]
        del self.targets[index]


    def set_image_size(self, image_size: int, greyscale: bool):
        self.image_size = image_size
        if greyscale:
            self.transform = v2.Compose(
                [
                    v2.Compose(
                        [
                            v2.ToImage(),
                            v2.ToDtype(torch.uint8, scale=True),
                            v2.ToDtype(torch.float32, scale=True),
                        ]
                    ),
                    v2.Grayscale(num_output_channels=1),
                    v2.Resize(image_size, max_size=image_size + 1, antialias=False),
                    v2.CenterCrop(image_size),
                ]
            )
        else:
            self.transform = v2.Compose(
                [
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                    v2.Resize(image_size, max_size=image_size + 1, antialias=False),
                    v2.CenterCrop(image_size),
                ]
            )

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True)
        CIFAR10(self.data_dir, train=False)

    def remove_train_example(self, idx):
        del self.cifar_train.indices[idx]

    def reset_train_data(self):
        self.cifar_train.indices = self.initial_train_indices.copy()

    def remove_train_data(self, indices_to_remove):
        self.cifar_train.indices = list(
            set(self.cifar_train.indices) - set(indices_to_remove)
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])
            cifar_full.set_indices(self.cifar_train.indices, self.cifar_val.indices)
            cifar_full.augments = self.augments

        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )
            self.cifar_test.set_indices([], range(len(self.cifar_test)))

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            num_workers=os.cpu_count(),
            shuffle=True,
            batch_size=BATCH_SIZE,
            persistent_workers=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            num_workers=os.cpu_count(),
            shuffle=False,
            batch_size=BATCH_SIZE,
            persistent_workers=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            num_workers=os.cpu_count(),
            shuffle=False,
            batch_size=BATCH_SIZE,
            persistent_workers=False,
        )


__all__ = ["CIFAR10DataModule"]

import os
import random
from typing import Any, Tuple
import albumentations as A
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10 as BaseDataset
from torchvision.transforms import v2

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")


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


class CIFAR10UBDataModule(L.LightningDataModule):
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

    task = "classification"

    def __init__(
        self,
        augmentation_policy_path,
        batch_size: int = 128,
        data_dir: str = PATH_DATASETS,
    ):
        super().__init__()

        assert os.path.exists(augmentation_policy_path)
        self.augments = A.load(augmentation_policy_path, data_format="yaml")
        self.data_dir = data_dir
        self.num_classes = 10
        self.batch_size = batch_size

        self.image_size = 224

        self.transform = v2.Compose(
            [
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
                v2.Resize(self.image_size, max_size=self.image_size + 1, antialias=False),
                v2.CenterCrop(self.image_size),
            ]
        )

    def reduce_samples_for_imbalance(self, dataset, reduce_classes: list[int], reduce_fraction: float):
        """Reduces the samples in the specified classes by the given fraction."""
        indices_to_keep = []
        for i, (img, target) in enumerate(zip(dataset.data, dataset.targets)):
            if target in reduce_classes:
                if random.random() < reduce_fraction:
                    indices_to_keep.append(i)
            else:
                indices_to_keep.append(i)
        
        dataset.data = dataset.data[indices_to_keep]
        dataset.targets = [dataset.targets[i] for i in indices_to_keep]

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            cifar_full = CIFAR10(self.data_dir, train=True, download = True, transform=self.transform)
            reduce_classes = [0, 1]  
            self.reduce_samples_for_imbalance(cifar_full, reduce_classes, reduce_fraction=0.2)
            dataset_length = len(cifar_full)

            train_size = int(0.9 * dataset_length)
            val_size = dataset_length - train_size

            self.cifar_train, self.cifar_val = random_split(cifar_full, [train_size, val_size])
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
            batch_size=self.batch_size,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            num_workers=os.cpu_count(),
            shuffle=False,
            batch_size=self.batch_size,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            num_workers=os.cpu_count(),
            shuffle=False,
            batch_size=self.batch_size,
            persistent_workers=True,
        )


__all__ = ["CIFAR10UBDataModule"]

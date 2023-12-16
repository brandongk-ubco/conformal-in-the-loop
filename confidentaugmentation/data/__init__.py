from .AugmentedCIFAR10 import AugmentedCIFAR10DataModule
from .AugmentedMNIST import AugmentedMNISTDataModule
from .CIFAR10 import CIFAR10DataModule
from .MNist import MNISTDataModule
from .AugmentedImageNet import AugmentedImageNetDataModule
from .ImageNet import ImageNetDataModule

__all__ = [
    "MNISTDataModule",
    "CIFAR10DataModule",
    "AugmentedCIFAR10DataModule",
    "AugmentedMNISTDataModule",
    "AugmentedImageNetDataModule",
    "ImageNetDataModule"
]

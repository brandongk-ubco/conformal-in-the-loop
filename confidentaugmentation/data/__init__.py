from .AugmentedCIFAR10 import AugmentedCIFAR10DataModule
from .AugmentedImageNet import AugmentedImageNetDataModule
from .AugmentedMNIST import AugmentedMNISTDataModule
from .CIFAR10 import CIFAR10DataModule
from .ImageNet import ImageNetDataModule
from .MNist import MNISTDataModule

__all__ = [
    "MNISTDataModule",
    "CIFAR10DataModule",
    "AugmentedCIFAR10DataModule",
    "AugmentedMNISTDataModule",
    "AugmentedImageNetDataModule",
    "ImageNetDataModule",
]

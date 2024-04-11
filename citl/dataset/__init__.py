from enum import Enum

from .CIFAR10 import CIFAR10DataModule
from .MNIST import MNISTDataModule


class Dataset(str, Enum):
    CIFAR10 = "CIFAR10"
    MNIST = "MNIST"

    @staticmethod
    def get(Dataset):
        if Dataset == "CIFAR10":
            return CIFAR10DataModule
        elif Dataset == "MNIST":
            return MNISTDataModule
        else:
            raise NotImplementedError(f"Dataset {Dataset} not implemented.")


__all__ = ["Dataset"]

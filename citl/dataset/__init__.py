from enum import Enum
from functools import partial

from .CIFAR10 import CIFAR10DataModule
from .Cityscapes import CityscapesDataModule
from .DFire import DFireDataModule
from .MNIST import MNISTDataModule
from .CelebA import CelebADataModule


class Dataset(str, Enum):
    CIFAR10 = "CIFAR10"
    MNIST = "MNIST"
    CityscapesCoarse = "CityscapesCoarse"
    CityscapesFine = "CityscapesFine"
    CelebA = 'CelebA'

    @staticmethod
    def get(Dataset):
        if Dataset == "CIFAR10":
            return CIFAR10DataModule
        elif Dataset == "DFire":
            return DFireDataModule
        elif Dataset == "MNIST":
            return MNISTDataModule
        elif Dataset == "CityscapesFine":
            return partial(CityscapesDataModule, train_mode="fine")
        elif Dataset == "CityscapesCoarse":
            return partial(CityscapesDataModule, train_mode="coarse")
        elif Dataset == "CelebA":
            return CelebADataModule
        else:
            raise NotImplementedError(f"Dataset {Dataset} not implemented.")


__all__ = ["Dataset"]

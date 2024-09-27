from enum import Enum
from functools import partial

from .CelebA import CelebADataModule
from .CIFAR10 import CIFAR10DataModule
from .CIFAR10UB import CIFAR10UBDataModule
from .Cityscapes import CityscapesDataModule
from .DFire import DFireDataModule
from .MNIST import MNISTDataModule
from .CIFAR10Noisy import CIFAR10NoisyDataModule


class Dataset(str, Enum):
    CIFAR10 = "CIFAR10"
    CIFAR10UB = "CIFAR10UB"
    CIFAR10Noisy = "CIFAR10Noisy"
    MNIST = "MNIST"
    CityscapesCoarse = "CityscapesCoarse"
    CityscapesFine = "CityscapesFine"
    CelebA = "CelebA"

    @staticmethod
    def get(Dataset):
        if Dataset == "CIFAR10":
            return CIFAR10DataModule
        elif Dataset == 'CIFAR10UB':
            return CIFAR10UBDataModule
        elif Dataset =='CIFAR10Noisy':
            return CIFAR10NoisyDataModule
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

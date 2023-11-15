from confidentaugmentation import cli
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as L
from .model import get_model
import os
import numpy as np

@cli.command()
def train():
    transform = transforms.ToTensor()
    train_set = MNIST(root="/workspaces/confident-augmentation/datasets/", download=True, train=True, transform=transform)
    val_set = MNIST(root="/workspaces/confident-augmentation/datasets/", download=True, train=False, transform=transform)

    train_loader = DataLoader(train_set, num_workers=os.cpu_count(), shuffle=True, pin_memory=True, batch_size=32, persistent_workers=True)
    val_loader = DataLoader(val_set, num_workers=os.cpu_count(), shuffle=True, pin_memory=True, batch_size=32, persistent_workers=True)

    trainer = L.Trainer()

    model = get_model([i for i in val_set])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

import os
import sys

import pytorch_lightning as L
import torch
from loguru import logger
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from confidentaugmentation import cli
from confidentaugmentation.control import PID
from confidentaugmentation.data import AugmentedMNISTDataModule
import torch.nn as nn
import torch.nn.functional as F

from .model.SimpleTrainer import SimpleTrainer

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@cli.command()
def simpletrain(
    augmentation_policy_path: str = "./policies/mnist.yaml",
    lr_method: str = "plateau",
    lr: float = 1e-3,
    optimizer: str = "Adam",
):
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True, warn_only=True)

    dm = AugmentedMNISTDataModule(augmentation_policy_path)

    model = SimpleTrainer(
        Net(),
        num_classes=dm.num_classes,
        lr_method=lr_method,
        lr=lr,
        optimizer=optimizer,
    )

    model_checkpoint = ModelCheckpoint(
        filename="{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=20,
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        model_checkpoint,
        early_stopping
    ]

    save_dir = os.path.join(
        "simple_logs",
        lr_method,
        optimizer,
    )

    trainer_logger = TensorBoardLogger(
        save_dir=save_dir,
        name="mnist",
    )

    trainer = L.Trainer(
        logger=trainer_logger,
        num_sanity_val_steps=sys.maxsize,
        max_epochs=sys.maxsize,
        deterministic=True,
        callbacks=callbacks,
        log_every_n_steps=10,
        accumulate_grad_batches=3
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)
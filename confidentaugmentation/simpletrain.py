import os
import sys

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import TensorBoardLogger

from confidentaugmentation import cli
from confidentaugmentation.data import AugmentedCIFAR10DataModule

from .model.SimpleTrainer import SimpleTrainer


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv1.weight, mode="fan_in", nonlinearity="relu")
        self.norm1 = torch.nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 3, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv2.weight, mode="fan_in", nonlinearity="relu")
        self.norm2 = torch.nn.BatchNorm2d(3)

        self.conv3 = nn.Conv2d(3, 16, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv3.weight, mode="fan_in", nonlinearity="relu")
        self.norm3 = torch.nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv4.weight, mode="fan_in", nonlinearity="relu")
        self.norm4 = torch.nn.BatchNorm2d(16)

        self.conv5 = nn.Conv2d(16, 24, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv5.weight, mode="fan_in", nonlinearity="relu")
        self.norm5 = torch.nn.BatchNorm2d(24)
        self.conv6 = nn.Conv2d(24, 24, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv6.weight, mode="fan_in", nonlinearity="relu")
        self.norm6 = torch.nn.BatchNorm2d(24)

        self.conv7 = nn.Conv2d(24, 32, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv7.weight, mode="fan_in", nonlinearity="relu")
        self.norm7 = torch.nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 32, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv8.weight, mode="fan_in", nonlinearity="relu")
        self.norm8 = torch.nn.BatchNorm2d(32)

        self.conv9 = nn.Conv2d(32, 48, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv9.weight, mode="fan_in", nonlinearity="relu")
        self.norm9 = torch.nn.BatchNorm2d(48)
        self.conv10 = nn.Conv2d(48, 48, 3, padding=1, padding_mode="zeros")
        nn.init.kaiming_uniform_(self.conv10.weight, mode="fan_in", nonlinearity="relu")
        self.norm10 = torch.nn.BatchNorm2d(48)

        self.fc1 = nn.Linear(48, 10)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)

    def forward(self, x):
        dropout_rate = 0.05

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, p=dropout_rate, training=self.training)

        x = self.conv3(x)
        x = self.norm3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.norm4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, p=dropout_rate, training=self.training)

        x = self.conv5(x)
        x = self.norm5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.norm6(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, p=dropout_rate, training=self.training)

        x = self.conv7(x)
        x = self.norm7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = self.norm8(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, p=dropout_rate, training=self.training)

        x = self.conv9(x)
        x = self.norm9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = self.norm10(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout2d(x, p=dropout_rate, training=self.training)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        return x


@cli.command()
def simpletrain(
    augmentation_policy_path: str = "./policies/cifar10.32.yaml",
    lr: float = 1e-3,
):
    L.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True, warn_only=True)

    dm = AugmentedCIFAR10DataModule(augmentation_policy_path)

    model = SimpleTrainer(
        Net(),
        num_classes=dm.num_classes,
        lr=lr,
    )

    model_checkpoint = ModelCheckpoint(
        filename="{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # early_stopping = EarlyStopping(
    #     monitor="val_loss",
    #     mode="min",
    #     patience=20,
    # )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        model_checkpoint,
        # early_stopping,
    ]

    trainer_logger = TensorBoardLogger(save_dir="simple_logs", name="cifar10")

    trainer = L.Trainer(
        logger=trainer_logger,
        num_sanity_val_steps=sys.maxsize,
        max_epochs=200,
        deterministic=True,
        callbacks=callbacks,
        log_every_n_steps=9,
    )

    trainer.fit(model=model, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)

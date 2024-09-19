from statistics import mean

import numpy as np
import pandas as pd
import pytorch_lightning as L
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.jaccard import JaccardIndex

from ..ConformalClassifier import ConformalClassifier
from ..utils.visualize_segmentation import visualize_segmentation


class Segmenter(L.LightningModule):
    def __init__(self, model, num_classes, lr=1e-3, lr_method="plateau"):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.num_classes = num_classes

        self.accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="none", ignore_index=0
        )
        self.jaccard = JaccardIndex(
            task="multiclass", num_classes=num_classes, average="none", ignore_index=0
        )

        self.val_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="none", ignore_index=0
        )
        self.val_jaccard = JaccardIndex(
            task="multiclass", num_classes=num_classes, average="none", ignore_index=0
        )

        self.test_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="none", ignore_index=0
        )
        self.test_jaccard = JaccardIndex(
            task="multiclass", num_classes=num_classes, average="none", ignore_index=0
        )

        self.lr = lr
        self.pixel_dropout = 0.0
        self.lr_method = lr_method

    def forward(self, x):
        if x.dim() == 2:
            y_hat = self.model(x.unsqueeze(0).unsqueeze(0))
        elif x.dim() == 3:
            y_hat = self.model(x.unsqueeze(0))
        elif x.dim() == 4:
            y_hat = self.model(x)
        else:
            raise ValueError("Input must be 2, 3 or 4 dimensional")

        if isinstance(y_hat, tuple):
            y_hat, _ = y_hat

        return y_hat

    def on_train_epoch_start(self) -> None:
        self.jaccard.reset()
        self.accuracy.reset()

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y.long(), reduction="none").mean()

        accs = self.accuracy(y_hat, y)
        self.log("accuracy", torch.mean(accs[1:]))
        self.log_dict(
            dict(
                zip(
                    [f"accuracy_{c}" for c in self.trainer.datamodule.classes[1:]],
                    accs[1:],
                )
            ),
            on_step=True,
            on_epoch=False,
        )

        jacs = self.jaccard(y_hat, y)
        self.log("jaccard", torch.mean(jacs[1:]))
        self.log_dict(
            dict(
                zip(
                    [f"jaccard_{c}" for c in self.trainer.datamodule.classes[1:]],
                    jacs[1:],
                )
            ),
            on_step=True,
            on_epoch=False,
        )

        self.log("loss", loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_jaccard.reset()
        self.val_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        val_loss = F.cross_entropy(y_hat, y.long(), reduction="none").mean()
        accs = self.val_accuracy(y_hat, y)
        self.log("val_accuracy", torch.mean(accs[1:]))
        self.log_dict(
            dict(
                zip(
                    [f"val_accuracy_{c}" for c in self.trainer.datamodule.classes[1:]],
                    accs[1:],
                )
            ),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        jacs = self.val_jaccard(y_hat, y)
        self.log("val_jaccard", torch.mean(jacs[1:]))
        self.log_dict(
            dict(
                zip(
                    [f"val_jaccard_{c}" for c in self.trainer.datamodule.classes[1:]],
                    jacs[1:],
                )
            ),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.log("val_loss", val_loss, on_step=False, on_epoch=True)

    def on_test_epoch_start(self) -> None:
        self.test_jaccard.reset()
        self.test_accuracy.reset()

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        test_loss = F.cross_entropy(y_hat, y.long(), reduction="none").mean()

        accs = self.test_accuracy(y_hat, y)
        self.log("test_accuracy", torch.mean(accs[1:]))
        self.log_dict(
            dict(
                zip(
                    [f"test_accuracy_{c}" for c in self.trainer.datamodule.classes[1:]],
                    accs[1:],
                )
            ),
            on_step=False,
            on_epoch=True,
        )

        jacs = self.test_jaccard(y_hat, y)
        self.log("test_jaccard", torch.mean(jacs[1:]))
        self.log_dict(
            dict(
                zip(
                    [f"test_jaccard_{c}" for c in self.trainer.datamodule.classes[1:]],
                    jacs[1:],
                )
            ),
            on_step=False,
            on_epoch=True,
        )

        self.log("test_loss", test_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.lr * 0.1
        )

        scheduler = None

        if self.lr_method == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.2,
                patience=10,
                min_lr=1e-6,
                verbose=True,
            )
            interval = "epoch"

        if scheduler:
            return [optimizer], [
                {
                    "scheduler": scheduler,
                    "interval": interval,
                    "monitor": "val_loss",
                }
            ]

        return optimizer


__all__ = ["Segmenter"]

import numpy as np
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torchmetrics.classification.accuracy import Accuracy


class SimpleTrainer(L.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        lr=1e-3,
        lr_method="plateau",
        optimizer="Adam"
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.classes_ = range(num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=len(self.classes_))

        self.lr = lr
        self.lr_method = lr_method
        self.optimizer = optimizer

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

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)
        self.log("val_accuracy", self.accuracy)

        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)

        self.accuracy(y_hat, y)
        self.log("test_accuracy", self.accuracy, on_step=False, on_epoch=True)

        self.log("test_loss", test_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.lr * 0.1
        )


__all__ = ["SimpleTrainer"]

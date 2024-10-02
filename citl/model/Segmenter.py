import pytorch_lightning as L
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from torchmetrics.classification.jaccard import JaccardIndex

# from ..losses.FocalLoss import FocalLoss
# from ..losses.TverskyLoss import TverskyLoss
from ..utils.visualize_segmentation import visualize_segmentation


class Segmenter(L.LightningModule):
    def __init__(self, model, num_classes, lr=1e-3, lr_method="plateau"):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = torch.nn.Sequential(torch.nn.InstanceNorm2d(3), model)

        self.num_classes = num_classes
        self.jaccard = JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            average="none",
            ignore_index=0,
            zero_division=1.0,
        )

        self.val_jaccard = JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            average="none",
            ignore_index=0,
            zero_division=1.0,
        )
        self.test_jaccard = JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            average="none",
            ignore_index=0,
            zero_division=1.0,
        )

        self.lr = lr
        self.lr_method = lr_method
        # self.entropy_loss = FocalLoss(
        #     "multiclass", reduction="none", from_logits=True, ignore_index=0
        # )
        self.entropy_loss = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=0)
        # self.overlap_loss = TverskyLoss(from_logits=True)

    def loss(self, y_hat, y):
        # num_classes = y_hat.shape[1]
        # y_one_hot = F.one_hot(y.long(), num_classes=num_classes)
        # y_one_hot = y_one_hot.permute(0, 3, 1, 2)
        loss = self.entropy_loss(y_hat, y.long())[y != 0].mean()
        # loss += self.overlap_loss(
        #     y_hat[:, 1:, :, :].reshape(-1), y_one_hot[:, 1:, :, :].reshape(-1)
        # )
        # classwise = torch.zeros(
        #     self.num_classes,
        #     dtype=loss.dtype,
        #     device=loss.device,
        #     requires_grad=loss.requires_grad,
        # )
        # classwise = torch.scatter_add(classwise, 0, ground_truths, loss)
        # classwise /= y.numel()
        return loss

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

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        if self.current_epoch == 0:
            img, target = x[1, :, :, :], y[1]
            if img.ndim > 2:
                img = img.moveaxis(0, -1)
            img = img - img.min()
            img = img / img.max()
            fig = visualize_segmentation(
                img.detach().cpu(), self.num_classes, mask=target[1:].detach().cpu()
            )
            if type(self.trainer.logger) is TensorBoardLogger:
                self.logger.experiment.add_figure(
                    "example_image", fig, self.global_step
                )
            elif type(self.trainer.logger) is NeptuneLogger:
                self.logger.experiment["training/example_image"].append(fig)
            plt.close()

        y_hat = self(x)
        loss = self.loss(y_hat, y)

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

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        val_loss = self.loss(y_hat, y)

        self.val_jaccard.update(y_hat, y)
        self.log(
            "val_loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        jacs = self.val_jaccard.compute()
        self.log(
            "val_jaccard",
            torch.mean(jacs[1:]),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_dict(
            dict(
                zip(
                    [f"val_jaccard_{c}" for c in self.trainer.datamodule.classes[1:]],
                    jacs[1:],
                )
            ),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def on_test_epoch_start(self) -> None:
        self.test_jaccard.reset()

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        test_loss = self.loss(y_hat, y)

        self.test_jaccard.update(y_hat, y)

        self.log(
            "test_loss",
            test_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

    def on_test_epoch_end(self):
        jacs = self.test_jaccard.compute()
        self.log("test_jaccard", torch.mean(jacs[1:]))
        self.log_dict(
            dict(
                zip(
                    [f"test_jaccard_{c}" for c in self.trainer.datamodule.classes[1:]],
                    jacs[1:],
                )
            )
        )

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

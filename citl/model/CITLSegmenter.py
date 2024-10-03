import pandas as pd
import pytorch_lightning as L
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from torchmetrics.classification.jaccard import JaccardIndex

from ..ConformalClassifier import ConformalClassifier
from ..utils.visualize_segmentation import visualize_segmentation


class CITLSegmenter(L.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        selectively_backpropagate=False,
        alpha=0.10,
        lr=1e-3,
        lr_method="plateau",
        method="score",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.conformal_classifier = ConformalClassifier(method=method, ignore_index=0)

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

        self.selectively_backpropagate = selectively_backpropagate
        self.alpha = alpha
        self.lr = lr
        self.pixel_dropout = 0.0
        self.lr_method = lr_method
        self.weight_decay = 0.0
        self.method = method

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
        self.class_weights = dict(
            zip(range(self.num_classes), [0.0] * self.num_classes)
        )
        self.class_counts = dict(
            zip(range(self.num_classes), [1e-5] * self.num_classes)
        )

    def training_step(self, batch, batch_idx):
        x, y, _ = batch

        y_hat = self(x)

        self.conformal_classifier.reset()
        self.conformal_classifier.append(y_hat, y)
        _, uncertainty = self.conformal_classifier.measure_uncertainty(alpha=self.alpha)

        metrics = dict([(k, v.float().mean()) for k, v in uncertainty.items()])
        self.log_dict(
            metrics, on_step=True, on_epoch=False, prog_bar=False, logger=True
        )

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

        if self.selectively_backpropagate:
            prediction_set_size = uncertainty["prediction_set_size"]
            loss = F.cross_entropy(y_hat, y.long(), reduction="none")[y != 0].flatten()
            loss_weights = prediction_set_size
            loss = loss * loss_weights
            loss = loss.mean()

            y_flt = y[y != 0].long()
            y_flt = y_flt.flatten()
            for clazz in range(1, self.num_classes):
                class_idxs = y_flt == clazz
                count = class_idxs.sum()
                weights = loss_weights[class_idxs].sum()
                self.class_counts[clazz] += count
                self.class_weights[clazz] += weights
        else:
            loss = F.cross_entropy(y_hat, y.long(), reduction="none")[y != 0].mean()

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

    def on_train_epoch_end(self) -> None:
        plt.figure()

        weights = {}

        for k in range(1, self.num_classes):
            label = self.trainer.datamodule.classes[k]
            weight = float(self.class_weights[k] / self.class_counts[k])
            weights[label] = weight
            self.log(
                f"mean_weight_{label}",
                weight,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"count_{label}",
                self.class_counts[k],
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            self.log(
                f"weight_{label}",
                self.class_weights[k],
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

        weight_max = max(weights.values())
        weight_min = min(weights.values())
        weight_range = max(weights.values()) - min(weights.values())
        self.log("weight_max", weight_max, on_step=False, on_epoch=True, sync_dist=True)
        self.log("weight_min", weight_min, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "weight_range", weight_range, on_step=False, on_epoch=True, sync_dist=True
        )

        weights_df = pd.DataFrame([weights]).T
        weights_df = weights_df.reset_index()

        weights_df = weights_df.rename(columns={"index": "class", 0: "mean_weight"})

        sns_plot = sns.barplot(data=weights_df, x="class", y="mean_weight")

        sns_plot.set_title(
            f"Mean Weighting of Each Class (epoch: {self.current_epoch + 1})"
        )
        sns_plot.set_xlabel(f"Class")
        sns_plot.set_ylabel("Mean Weighting")
        plt.xticks(rotation=90)
        plt.tight_layout()

        if type(self.trainer.logger) is TensorBoardLogger:
            self.logger.experiment.add_figure(
                "mean_class_weights",
                sns_plot.get_figure(),
                self.global_step,
            )

        elif type(self.trainer.logger) is NeptuneLogger:
            self.logger.experiment["training/mean_class_weights"].append(
                sns_plot.get_figure()
            )
        plt.close()

    def on_validation_epoch_start(self) -> None:
        self.val_jaccard.reset()
        self.conformal_classifier.reset()
        self.val_batch_idx_fit_uncertainty = (
            len(self.trainer.datamodule.val_dataloader()) // 10
        )

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        val_loss = F.cross_entropy(y_hat, y.long(), reduction="none")[y != 0].mean()

        if batch_idx < self.val_batch_idx_fit_uncertainty:
            self.conformal_classifier.append(y_hat, y, percentage=0.1)
        elif batch_idx == self.val_batch_idx_fit_uncertainty:
            self.conformal_classifier.fit(alphas=set([self.alpha]))
        else:
            self.conformal_classifier.append(y_hat, y)
            _, uncertainty = self.conformal_classifier.measure_uncertainty(
                alpha=self.alpha
            )

            metrics = dict(
                [(f"val_{k}", v.float().mean()) for k, v in uncertainty.items()]
            )
            self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True)

        self.val_jaccard.update(y_hat, y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        jacs = self.val_jaccard.compute()
        self.log(
            "val_jaccard",
            torch.mean(jacs[1:]),
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            prog_bar=True,
        )
        self.log_dict(
            dict(
                zip(
                    [f"val_jaccard_{c}" for c in self.trainer.datamodule.classes[1:]],
                    jacs[1:],
                )
            ),
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

        quantiles = self.conformal_classifier.quantiles
        quantiles = {
            f"quantile_{k}": v.detach().cpu().numpy().tolist()
            for k, v in quantiles.items()
        }
        self.log_dict(
            quantiles, prog_bar=False, on_epoch=True, on_step=False, sync_dist=True
        )

    def on_test_epoch_start(self) -> None:
        self.test_jaccard.reset()

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        test_loss = F.cross_entropy(y_hat, y.long(), reduction="none")[y != 0].mean()

        self.conformal_classifier.reset()
        self.conformal_classifier.append(y_hat, y)
        _, uncertainty = self.conformal_classifier.measure_uncertainty(alpha=self.alpha)

        metrics = dict(
            [(f"test_{k}", v.float().mean()) for k, v in uncertainty.items()]
        )
        self.log_dict(metrics, on_epoch=True, on_step=False, sync_dist=True)

        self.test_jaccard.update(y_hat, y)

        self.log("test_loss", test_loss, on_epoch=True, on_step=False, sync_dist=True)

    def on_test_epoch_end(self):
        jacs = self.test_jaccard.compute()
        self.log(
            "test_jaccard",
            torch.mean(jacs[1:]),
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log_dict(
            dict(
                zip(
                    [f"test_jaccard{c}" for c in self.trainer.datamodule.classes[1:]],
                    jacs[1:],
                )
            ),
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.lr * 0.1
        )

        scheduler = None

        if self.lr_method == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
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
                    "monitor": "val_jaccard",
                }
            ]

        return optimizer


__all__ = ["CITLSegmenter"]

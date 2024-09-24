import pandas as pd
import pytorch_lightning as L
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pytorch_lightning.loggers import NeptuneLogger, TensorBoardLogger
from torchmetrics.classification.accuracy import Accuracy

from ..ConformalClassifier import ConformalClassifier
from ..losses.FocalLoss import FocalLoss


class CITLClassifier(L.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        selectively_backpropagate=False,
        alpha=0.10,
        val_alpha=0.10,
        lr=1e-3,
        lr_method="plateau",
        method="score",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = torch.nn.Sequential(torch.nn.InstanceNorm2d(3), model)

        self.conformal_classifier = ConformalClassifier(method=method)

        self.num_classes = num_classes

        self.accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )
        self.test_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )
        self.val_accuracy = Accuracy(
            task="multiclass", num_classes=num_classes, average="none"
        )

        self.selectively_backpropagate = selectively_backpropagate
        self.alpha = alpha
        self.val_alpha = val_alpha
        self.lr = lr
        self.pixel_dropout = 0.0
        self.lr_method = lr_method
        self.weight_decay = 0.0
        self.method = method
        self.examples_without_uncertainty = {}
        self.test_results = []
        self.loss = FocalLoss(reduction="none")

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

    def on_train_start(self) -> None:
        self.initial_train_set = self.trainer.train_dataloader.dataset
        self.initial_train_size = float(len(self.trainer.train_dataloader.dataset))
        self.conformal_classifier.reset()

    def on_train_epoch_start(self) -> None:
        self.accuracy.reset()
        current_train_size = float(len(self.trainer.train_dataloader.dataset))
        self.log("Train Dataset Size", current_train_size / self.initial_train_size)
        self.class_weights = dict(
            zip(range(self.num_classes), [0.0] * self.num_classes)
        )
        self.class_counts = dict(
            zip(range(self.num_classes), [1e-5] * self.num_classes)
        )

    def training_step(self, batch, batch_idx):
        x, y, indeces = batch

        y_hat = self(x)

        if self.current_epoch == 0:
            img, target = x[1, :, :, :], y[1]
            if img.ndim > 2:
                img = img.moveaxis(0, -1)
            img = img - img.min()
            img = img / img.max()
            label = self.trainer.datamodule.classes[target]
            fig = plt.figure()
            plt.imshow(img.detach().cpu().numpy())
            plt.title(label)
            plt.axis("off")
            if type(self.trainer.logger) is TensorBoardLogger:
                self.logger.experiment.add_figure(
                    "example_image", fig, self.global_step
                )
            elif type(self.trainer.logger) is NeptuneLogger:
                self.logger.experiment["training/example_image"].append(fig)
            plt.close()

        self.conformal_classifier.reset()
        self.conformal_classifier.append(y_hat, y)
        _, uncertainty = self.conformal_classifier.measure_uncertainty(alpha=self.alpha)

        metrics = dict([(k, v.float().mean()) for k, v in uncertainty.items()])
        self.log_dict(
            metrics, on_step=True, on_epoch=False, prog_bar=False, logger=True
        )

        uncertain = uncertainty["uncertain"]

        for i, idx in enumerate(indeces.detach().cpu().numpy()):
            if uncertain[i]:
                self.examples_without_uncertainty[idx] = 0
            else:
                if idx in self.examples_without_uncertainty:
                    self.examples_without_uncertainty[idx] += 1
                else:
                    self.examples_without_uncertainty[idx] = 1

        loss = self.loss(y_hat, y)
        if self.selectively_backpropagate:
            prediction_set_size = uncertainty["prediction_set_size"].reshape(y.shape)
            loss_weights = prediction_set_size
            loss = loss * loss_weights

            y_flt = y.flatten()
            p_flt = loss_weights.flatten()
            for clazz in range(self.num_classes):
                class_idxs = y_flt == clazz
                count = class_idxs.sum()
                weights = p_flt[class_idxs].sum()
                self.class_counts[clazz] += count
                self.class_weights[clazz] += weights

        loss = loss.mean()

        accs = self.accuracy(y_hat, y)
        self.log("accuracy", torch.mean(accs))
        self.log_dict(
            dict(
                zip(
                    [f"accuracy_{c}" for c in self.trainer.datamodule.classes],
                    accs,
                )
            ),
        )

        self.log("loss", loss)
        return loss

    def on_train_epoch_end(self) -> None:
        plt.figure()

        weights = {}

        for k in range(self.num_classes):
            label = self.trainer.datamodule.classes[k]
            weight = float(self.class_weights[k] / self.class_counts[k])
            weights[label] = weight
            self.log(f"mean_weight_{label}", weight, on_step=False, on_epoch=True)
            self.log(
                f"count_{label}",
                self.class_counts[k],
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                f"weight_{label}",
                self.class_weights[k],
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        weight_max = max(weights.values())
        weight_min = min(weights.values())
        weight_range = max(weights.values()) - min(weights.values())
        self.log("weight_max", weight_max, on_step=False, on_epoch=True, logger=True)
        self.log("weight_min", weight_min, on_step=False, on_epoch=True, logger=True)
        self.log(
            "weight_range", weight_range, on_step=False, on_epoch=True, logger=True
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
        self.val_accuracy.reset()
        self.conformal_classifier.reset()
        self.val_batch_idx_fit_uncertainty = (
            len(self.trainer.datamodule.val_dataloader()) // 5
        )

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        val_loss = self.loss(y_hat, y).mean()

        if batch_idx < self.val_batch_idx_fit_uncertainty:
            self.conformal_classifier.append(y_hat, y)
        elif batch_idx == self.val_batch_idx_fit_uncertainty:
            self.conformal_classifier.fit(alphas=set([self.alpha, self.val_alpha]))
            quantiles = self.conformal_classifier.quantiles
            quantiles = {
                f"quantile_{k}": v.detach().cpu().numpy().tolist()
                for k, v in quantiles.items()
            }
            self.log_dict(quantiles, prog_bar=False)
        else:
            self.conformal_classifier.append(y_hat, y)
            _, uncertainty = self.conformal_classifier.measure_uncertainty(
                alpha=self.val_alpha
            )

            metrics = dict(
                [(f"val_{k}", v.float().mean()) for k, v in uncertainty.items()]
            )
            self.log_dict(metrics, prog_bar=True)

        self.val_accuracy.update(y_hat, y)
        self.log("val_loss", val_loss)

    def on_validation_epoch_end(self):
        accs = self.val_accuracy.compute()
        self.log("val_accuracy", torch.mean(accs), prog_bar=True)
        self.log_dict(
            dict(
                zip(
                    [f"val_accuracy_{c}" for c in self.trainer.datamodule.classes],
                    accs,
                )
            ),
        )

    # def test_step(self, batch, batch_idx):
    #     x, y, _ = batch
    #     y_hat = self(x)

    #     test_loss = F.cross_entropy(y_hat, y)

    #     self.conformal_classifier.append(y_hat, y)
    #     conformal_sets, uncertainty = self.conformal_classifier.measure_uncertainty(
    #         alpha=self.val_alpha
    #     )

    #     metrics = dict(
    #         [(f"test_{k}", v.float().mean()) for k, v in uncertainty.items()]
    #     )
    #     self.log_dict(
    #         metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True
    #     )

    #     for idx in range(x.shape[0]):
    #         img, target = x[idx, :, :, :], y[idx]
    #         if img.ndim > 2:
    #             img = img.moveaxis(0, -1)
    #         img = img - img.min()
    #         img = img / img.max()
    #         expected = self.trainer.datamodule.classes[target]

    #         combined = list(zip(y_hat[idx], conformal_sets[idx], self.trainer.datamodule.classes))
    #         filtered = [tup for tup in combined if tup[1]]

    #         sorted_filtered = sorted(filtered, key=lambda x: -x[0])
    #         labels = set([x[2] for x in sorted_filtered])
    #         labels = str(labels).replace("'", "")

    #         set_size = uncertainty["prediction_set_size"][idx].item()
    #         correct = y_hat[idx].argmax().item() == target

    #         if set_size == 0:
    #             example_type = "atypical"
    #         elif set_size == 1 and correct:
    #             example_type = "realized"
    #         elif set_size == 1 and not correct:
    #             example_type = "confused"
    #         elif set_size > 1:
    #             example_type = "uncertain"
    #         else:
    #             raise ValueError(f"Unknown Set Size {set_size}")

    #         fig = plt.figure()
    #         plt.imshow(img.detach().cpu().numpy())
    #         plt.title(f"Ground Truth: {expected}")
    #         plt.suptitle(f"Predicted: {labels}")
    #         plt.axis("off")
    #         if type(self.trainer.logger) is TensorBoardLogger:
    #             self.logger.experiment.add_figure(
    #                 f"test_images_{example_type}", fig, self.global_step
    #             )
    #         elif type(self.trainer.logger) is NeptuneLogger:
    #             self.logger.experiment[f"test/example_images/{example_type}"].append(fig)
    #         plt.close()

    #     self.accuracy(y_hat, y)
    #     self.log("test_accuracy", self.accuracy, on_step=False, on_epoch=True)

    #     self.log("test_loss", test_loss, on_step=False, on_epoch=True)

    def on_test_epoch_start(self):
        self.test_accuracy.reset()

    def test_step(self, batch, batch_idx):
        x, y, attributes = batch
        y_hat = self(x)

        test_loss = self.loss(y_hat, y).mean()

        self.conformal_classifier.append(y_hat, y)
        _, uncertainty = self.conformal_classifier.measure_uncertainty(
            alpha=self.val_alpha
        )

        metrics = dict(
            [(f"test_{k}", v.float().mean()) for k, v in uncertainty.items()]
        )
        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        probabilities = torch.softmax(y_hat, dim=1)
        for i in range(len(y)):
            self.test_results.append(
                {
                    "label": y[i].item(),
                    "attribute": attributes[i].item(),
                    "probability": probabilities[i].tolist(),
                }
            )

        self.test_accuracy.update(y_hat, y)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        df = pd.DataFrame(self.test_results)
        df.to_csv("test_results.csv", index=False)
        self.test_results = []

        accs = self.test_accuracy.compute()
        self.log("test_accuracy", torch.mean(accs))
        self.log_dict(
            dict(
                zip(
                    [f"test_accuracy_{c}" for c in self.trainer.datamodule.classes],
                    accs,
                )
            ),
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
                min_lr=1e-7,
                verbose=True,
            )
            interval = "epoch"

        if scheduler:
            return [optimizer], [
                {
                    "scheduler": scheduler,
                    "interval": interval,
                    "monitor": "val_accuracy",
                }
            ]

        return optimizer


__all__ = ["CITLClassifier"]

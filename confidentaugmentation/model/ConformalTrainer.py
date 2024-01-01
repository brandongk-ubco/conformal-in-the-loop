import numpy as np
import pytorch_lightning as L
import seaborn as sns
import torch
import torch.nn.functional as F
from mapie.classification import MapieClassifier
from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchvision.transforms import v2


class ConformalTrainer(L.LightningModule):
    def __init__(
        self,
        model,
        num_classes,
        selectively_backpropagate=False,
        mapie_alpha=0.10,
        val_mapie_alpha=0.10,
        warmup_epochs=3,
        lr=1e-3,
        pid=None,
        lr_method="plateau",
        optimizer="Adam",
        control_weight_decay=False,
        control_pixel_dropout=False,
        mapie_method="score",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        self.classes_ = range(num_classes)

        self.accuracy = Accuracy(task="multiclass", num_classes=len(self.classes_))

        self.atypical_percentage = MeanMetric()
        self.uncertain_percentage = MeanMetric()
        self.confused_percentage = MeanMetric()
        self.realized_percentage = MeanMetric()

        self.selectively_backpropagate = selectively_backpropagate
        self.mapie_alpha = mapie_alpha
        self.val_mapie_alpha = val_mapie_alpha
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.pid = pid
        self.pixel_dropout = 0.0
        self.lr_method = lr_method
        self.optimizer = optimizer
        self.control_weight_decay = control_weight_decay
        self.control_pixel_dropout = control_pixel_dropout
        self.weight_decay = 0.0
        self.mapie_method = mapie_method
        self.examples_without_uncertainty = {}

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, x, y):
        raise NotImplementedError("Cannot fit this way.")

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    def predict_proba(self, x):
        return np.array([v[0] for v in self.cp_examples])[x]

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
        super().on_train_epoch_start()
        self.atypical_percentage.reset()
        self.uncertain_percentage.reset()
        self.confused_percentage.reset()
        self.realized_percentage.reset()

    def training_step(self, batch, batch_idx):
        x, y, indeces = batch

        if self.control_pixel_dropout:
            minimum = x.min()
            x = x - minimum
            torch.nn.functional.dropout(x, p=self.pixel_dropout, inplace=True)
            x = x + minimum

        if self.control_weight_decay:
            self.optimizers().optimizer.param_groups[0][
                "weight_decay"
            ] = self.weight_decay

        y_hat = self(x)

        if self.current_epoch == 1:
            img, target = x[1, :, :, :], y[1]
            img = img - img.min()
            img = img / img.max()
            label = self.trainer.datamodule.classes[target]
            self.logger.experiment.add_image(f"{label}", img, self.global_step)

        self.cp_examples = list(
            zip(y_hat.detach().softmax(axis=1).cpu().numpy(), y.detach().cpu().numpy())
        )

        num_classes = (
            self.mapie_classifier.predict(
                range(len(self.cp_examples)), alpha=[self.mapie_alpha]
            )[1]
            .sum(axis=1)
            .squeeze()
        )

        predicted = y_hat.argmax(axis=1)

        correct = predicted == y

        atypical = torch.tensor(num_classes == 0).to(device=self.device)
        realized = torch.logical_and(
            correct, torch.tensor(num_classes == 1).to(device=self.device)
        )
        confused = torch.logical_and(
            ~correct, torch.tensor(num_classes == 1).to(device=self.device)
        )
        uncertain = torch.tensor(num_classes > 1).to(device=self.device)

        for i, idx in enumerate(indeces.detach().cpu().numpy()):
            if uncertain[i]:
                self.examples_without_uncertainty[idx] = 0
            else:
                if idx in self.examples_without_uncertainty:
                    self.examples_without_uncertainty[idx] += 1
                else:
                    self.examples_without_uncertainty[idx] = 1

        atypical_percentage = atypical.sum() / len(atypical)
        realized_percentage = realized.sum() / len(realized)
        confused_percentage = confused.sum() / len(confused)
        uncertain_percentage = uncertain.sum() / len(uncertain)

        self.atypical_percentage(atypical_percentage)
        self.realized_percentage(realized_percentage)
        self.confused_percentage(confused_percentage)
        self.uncertain_percentage(uncertain_percentage)

        torch.testing.assert_close(
            atypical_percentage
            + realized_percentage
            + confused_percentage
            + uncertain_percentage,
            torch.tensor(1.0).to(self.device),
        )

        metrics = {
            "atypical": self.atypical_percentage.compute(),
            "uncertain": self.uncertain_percentage.compute(),
            "confused": self.confused_percentage.compute(),
            "realized": self.realized_percentage.compute(),
        }
        self.log_dict(metrics, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if self.pid is not None:
            pid_value = float(self.pid(metrics["atypical"]))
            self.log(
                "pid_setpoint",
                self.pid.get_setpoint(),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
            )

            if self.control_pixel_dropout:
                self.pixel_dropout = pid_value / 10
                self.log(
                    "pixel_dropout",
                    self.pixel_dropout,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=True,
                    logger=True,
                )

            if self.control_weight_decay:
                self.weight_decay = pid_value / 1000

        self.log(
            "weight_decay",
            self.optimizers().optimizer.param_groups[0]["weight_decay"],
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

        if self.selectively_backpropagate:
            loss = F.cross_entropy(y_hat, y, reduction="none")[uncertain].mean()
        else:
            loss = F.cross_entropy(y_hat, y)

        self.log("loss", loss)
        return loss

    def on_train_epoch_end(self) -> None:
        examples_without_uncertainty = np.array(
            list(self.examples_without_uncertainty.values())
        )
        sns_plot = sns.histplot(
            data=examples_without_uncertainty,
            stat="percent",
            binwidth=1,
            binrange=(0, 20),
        )
        sns_plot.set_title(
            f"Histogram of examples without uncertainty (epoch: {self.current_epoch}, mean: {examples_without_uncertainty.mean():.2f})"
        )

        self.logger.experiment.add_figure(
            "examples_without_uncertainty",
            sns_plot.get_figure(),
            self.global_step,
        )

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch

        y_hat = self(x)
        if isinstance(y_hat, tuple):
            y_hat, _ = y_hat

        test_loss = F.cross_entropy(y_hat, y)

        self.cp_examples += list(
            zip(y_hat.detach().softmax(axis=1).cpu().numpy(), y.detach().cpu().numpy())
        )

        self.val_labels += list(y.detach().cpu().numpy())

        self.accuracy(y_hat, y)
        self.log("val_accuracy", self.accuracy, on_step=False, on_epoch=True)

        self.log("val_loss", test_loss, on_step=False, on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.accuracy.reset()
        self.cp_examples = []
        self.val_labels = []

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

        self.mapie_classifier = MapieClassifier(
            estimator=self, method=self.mapie_method, cv="prefit", n_jobs=-1
        ).fit(np.array(range(len(self.cp_examples))), [v[1] for v in self.cp_examples])

        conformal_sets = self.mapie_classifier.predict(
            range(len(self.cp_examples)), alpha=[self.val_mapie_alpha]
        )[1]

        num_classes = conformal_sets.sum(axis=1).squeeze()

        conformal_predictions = conformal_sets.argmax(axis=1).squeeze()

        correct = conformal_predictions == self.val_labels

        atypical = num_classes == 0
        realized = np.logical_and(correct, num_classes == 1)
        confused = np.logical_and(~correct, num_classes == 1)
        uncertain = num_classes > 1

        atypical_percentage = atypical.sum() / len(atypical)
        realized_percentage = realized.sum() / len(realized)
        confused_percentage = confused.sum() / len(confused)
        uncertain_percentage = uncertain.sum() / len(uncertain)

        metrics = {
            "val_atypical": atypical_percentage,
            "val_uncertain": uncertain_percentage,
            "val_confused": confused_percentage,
            "val_realized": realized_percentage,
        }
        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        # if self.pid:
        #     self.pid.set_setpoint(metrics["val_realized"])

        if self.lr_method == "uncertainty":
            self.optimizers().optimizer.param_groups[0]["lr"] = (
                self.lr * 0.9 * metrics["val_uncertain"] + self.lr * 0.1
            )

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = F.cross_entropy(y_hat, y)

        conformal_sets = self.mapie_classifier.predict(
            range(len(self.cp_examples)), alpha=[self.val_mapie_alpha]
        )[1]

        num_classes = conformal_sets.sum(axis=1).squeeze()

        conformal_predictions = conformal_sets.argmax(axis=1).squeeze()

        correct = conformal_predictions == self.val_labels

        atypical = num_classes == 0
        realized = np.logical_and(correct, num_classes == 1)
        confused = np.logical_and(~correct, num_classes == 1)
        uncertain = num_classes > 1

        atypical_percentage = atypical.sum() / len(atypical)
        realized_percentage = realized.sum() / len(realized)
        confused_percentage = confused.sum() / len(confused)
        uncertain_percentage = uncertain.sum() / len(uncertain)

        metrics = {
            "test_atypical": atypical_percentage,
            "test_uncertain": uncertain_percentage,
            "test_confused": confused_percentage,
            "test_realized": realized_percentage,
        }

        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )

        self.accuracy(y_hat, y)
        self.log("test_accuracy", self.accuracy, on_step=False, on_epoch=True)

        self.log("test_loss", test_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        dataloader = self.trainer.datamodule.train_dataloader()

        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.lr * 0.1,
                nesterov=True,
            )
        elif self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.lr * 0.1
            )
        else:
            raise NotImplementedError("Optimizer not implemented.")

        scheduler = None

        if self.lr_method == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr * 10,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=len(dataloader),
                anneal_strategy="cos",
                pct_start=self.warmup_epochs / self.trainer.max_epochs,
                cycle_momentum=False,
                div_factor=10,
                final_div_factor=100,
                three_phase=True,
            )
            interval = "step"

        if self.lr_method == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max" if self.selectively_backpropagate else "min",
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
                    "monitor": "val_realized"
                    if self.selectively_backpropagate
                    else "val_loss",
                }
            ]

        return optimizer


__all__ = ["ConformalTrainer"]

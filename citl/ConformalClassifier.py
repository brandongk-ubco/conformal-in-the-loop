import numpy as np
import torch
from loguru import logger
from mapie.classification import MapieClassifier
import random

class ConformalClassifier:

    def __init__(self, mapie_method="aps"):
        self.mapie_method = mapie_method
        self.mapie_classifier = None
        self.reset()

    def __sklearn_is_fitted__(self):
        return True

    def reset(self):
        self.cp_examples = []
        self.val_labels = []

    def append(self, y_hat, y, percent=None):
        if y_hat.ndim > 2:
            y_hat = y_hat.moveaxis(1, -1).flatten(end_dim=y_hat.ndim - 2)

        if y.ndim > 1:
            y = y.flatten()

        if percent is not None:
            probs = torch.empty(len(y)).uniform_()
            idx = (probs < percent).nonzero().flatten()
            y = y[idx]
            y_hat = y_hat[idx, :]

        if torch.is_tensor(y_hat):
            y_hat = y_hat.softmax(axis=1).detach().cpu().numpy()
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()

        assert y.ndim == 1
        assert y_hat.ndim == 2

        self.cp_examples.append(y_hat)
        self.val_labels.append(y)

    def fit(self):
        self.cp_examples = np.concatenate(self.cp_examples, axis=0)
        self.val_labels = np.concatenate(self.val_labels, axis=0)

        self.classes_ = range(self.cp_examples.shape[1])
        num_examples = len(self.cp_examples)

        examples = range(num_examples)
        labels = self.val_labels
        if num_examples > 1e8:
            examples = random.sample(examples, int(1e8))
            labels = self.val_labels[examples]

        self.mapie_classifier = MapieClassifier(
            estimator=self, method=self.mapie_method, cv="prefit", n_jobs=-1
        ).fit(
            np.array(examples),
            labels,
        )

        self.cp_examples = []
        self.val_labels = []

    def measure_uncertainty(self, alphas=[0.1]):
        self.cp_examples = np.concatenate(self.cp_examples, axis=0)
        self.val_labels = np.concatenate(self.val_labels, axis=0)

        self.classes_ = range(self.cp_examples.shape[1])
        num_examples = len(self.cp_examples)

        if self.mapie_classifier is None:
            logger.warning(
                "MAPIE model not fit, fitting now.  Uncertainty may be inaccurate."
            )
            self.mapie_classifier = MapieClassifier(
                estimator=self, method=self.mapie_method, cv="prefit", n_jobs=-1
            ).fit(
                np.array(range(num_examples)),
                self.val_labels,
            )

        conformal_sets = self.mapie_classifier.predict(
            range(num_examples), alpha=alphas
        )[1]

        num_classes = conformal_sets.sum(axis=1).squeeze()
        conformal_predictions = conformal_sets.argmax(axis=1).squeeze()
        correct = conformal_predictions == self.val_labels

        atypical = num_classes == 0
        realized = np.logical_and(correct, num_classes == 1)
        confused = np.logical_and(~correct, num_classes == 1)
        uncertain = num_classes > 1

        results = {
            "prediction_set_size": num_classes,
            "atypical": atypical,
            "realized": realized,
            "confused": confused,
            "uncertain": uncertain,
        }

        self.val_labels = []
        self.cp_examples = []

        return conformal_sets, results

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    def predict_proba(self, x):
        return self.cp_examples[x]


__all__ = ["ConformalClassifier"]

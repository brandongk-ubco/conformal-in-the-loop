import numpy as np
import torch
from mapie.classification import MapieClassifier


class ConformalClassifier:

    def __init__(self, mapie_method="aps"):
        self.mapie_method = mapie_method
        self.mapie_classifier = None
        self.cp_examples = None
        self.val_labels = None

    def __sklearn_is_fitted__(self):
        return True

    def reset(self):
        self.cp_examples = None
        self.val_labels = None

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

        if self.cp_examples is None:
            self.cp_examples = y_hat
        else:
            self.cp_examples = np.row_stack([self.cp_examples, y_hat])

        if self.val_labels is None:
            self.val_labels = y
        else:
            self.val_labels = np.append(self.val_labels, y)

        assert self.cp_examples.shape[0] == len(self.val_labels)

    def fit(self, percentage=1.0):
        num_available = len(self.val_labels)
        use_idx = int(num_available * percentage)

        y = self.val_labels[:use_idx]
        y_hat = self.cp_examples[:use_idx]

        num_examples = len(y)

        assert y_hat.shape[0] == num_examples
        self.classes_ = range(y_hat.shape[1])

        self.mapie_classifier = MapieClassifier(
            estimator=self, method=self.mapie_method, cv="prefit", n_jobs=-1
        ).fit(
            np.array(range(use_idx)),
            y,
        )

    def measure_uncertainty(self, percentage=1.0, alphas=[0.1]):
        num_available = len(self.val_labels)
        use_idx = num_available - int(num_available * percentage)

        y = self.val_labels[use_idx:]

        conformal_sets = self.mapie_classifier.predict(
            range(use_idx, num_available), alpha=alphas
        )[1]

        num_classes = conformal_sets.sum(axis=1).squeeze()
        conformal_predictions = conformal_sets.argmax(axis=1).squeeze()
        correct = conformal_predictions == y

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
        return conformal_sets, results

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    def predict_proba(self, x):
        return self.cp_examples[x]


__all__ = ["ConformalClassifier"]

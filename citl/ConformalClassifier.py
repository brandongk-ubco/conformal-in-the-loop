from functools import partial

import numpy as np
import torch


def lac(y_hat, y):
    return 1 - y_hat[y.int()]


def lac_set(y_hat, quantile):
    return y_hat >= 1 - quantile


def reduce_score(example, function):
    y_hat = example[1:]
    y = example[:1]
    return function(y_hat, y)


def reduce_quantile(y_hat, function, quantile):
    return function(y_hat, quantile)


def reduce_correct(example):
    y_hat = example[1:]
    y = example[:1]
    return y_hat[y.int()].bool()


class ConformalClassifier:

    def __init__(self, method="score"):
        self.method = method
        self.reset()

    def reset(self):
        self.cp_examples = []
        self.val_labels = []

    def append(self, y_hat, y):
        if y_hat.ndim > 2:
            y_hat = y_hat.moveaxis(1, -1).flatten(end_dim=y_hat.ndim - 2)

        if y.ndim > 1:
            y = y.flatten()

        if torch.is_tensor(y_hat):
            y_hat = y_hat.softmax(axis=1).detach()
        if torch.is_tensor(y):
            y = y.detach()

        assert y.ndim == 1
        assert y_hat.ndim == 2

        self.cp_examples.append(y_hat)
        self.val_labels.append(y)

    def fit(self, alphas=[0.1]):
        self.cp_examples = torch.concatenate(self.cp_examples, axis=0)
        self.val_labels = torch.concatenate(self.val_labels, axis=0)

        examples = torch.cat((self.val_labels.unsqueeze(1), self.cp_examples), axis=1)
        mapper = partial(reduce_score, function=lac)

        scores = torch.vmap(mapper)(examples).squeeze()

        num = scores.size()[0]
        self.quantiles = {}
        for alpha in alphas:
            quantile = torch.quantile(scores, (num + 1) * (1 - alpha) / num)
            self.quantiles[alpha] = quantile

        self.cp_examples = []
        self.val_labels = []


    def measure_uncertainty(self, alpha=0.1):
        self.cp_examples = torch.concatenate(self.cp_examples, axis=0)
        self.val_labels = torch.concatenate(self.val_labels, axis=0)

        quantile = self.quantiles[alpha]

        mapper = partial(reduce_quantile, function=lac_set, quantile=quantile)

        conformal_sets = torch.vmap(mapper)(self.cp_examples).squeeze()

        correct = self.cp_examples.argmax(axis=1) == self.val_labels

        num_classes = conformal_sets.sum(axis=1)

        atypical = num_classes == 0
        realized = torch.logical_and(correct, num_classes == 1)
        confused = torch.logical_and(~correct, num_classes == 1)
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


__all__ = ["ConformalClassifier"]

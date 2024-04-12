from mapie.classification import MapieClassifier
import numpy as np

class ConformalClassifier():

    def __init__(self, mapie_method="aps"):
        self.mapie_method = mapie_method
        self.cp_examples = []
        self.mapie_classifier = None

    def __sklearn_is_fitted__(self):
        return True
    
    def reset(self):
        self.cp_examples = []

    def append(self, y_hat, y):
        self.cp_examples += y_hat.detach().softmax(axis=1).cpu().numpy()
        self.val_labels += y.detach().cpu().numpy()
    
    def fit(self, y_hat, y):
        num_examples = len(y)

        assert y_hat.shape[0] == y
        
        self.cp_examples = y_hat

        self.mapie_classifier = MapieClassifier(
            estimator=self, method=self.mapie_method, cv="prefit", n_jobs=-1
        ).fit(
            np.array(range(num_examples)),
            [v[1] for i, v in y],
        )

    def estimate_uncertainty(self, alphas):
        conformal_sets = self.mapie_classifier.predict(
            range(len(self.cp_examples)), alpha=alphas
        )[1]

        num_classes = conformal_sets.sum(axis=1).squeeze()

        conformal_predictions = conformal_sets.argmax(axis=1).squeeze()

        correct = conformal_predictions == self.val_labels[calib_idx:]

        atypical = num_classes == 0
        realized = np.logical_and(correct, num_classes == 1)
        confused = np.logical_and(~correct, num_classes == 1)
        uncertain = num_classes > 1

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    def predict_proba(self, x):
        return self.cp_examples[x]


__all__ = ["ConformalClassifier"]
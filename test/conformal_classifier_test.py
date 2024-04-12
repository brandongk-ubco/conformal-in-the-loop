from citl.ConformalClassifier import ConformalClassifier
import numpy as np
import torch

class TestConformalClassifier:
    def test_initialize(self):
        cc = ConformalClassifier()
        assert cc.__sklearn_is_fitted__() == True
        assert cc.cp_examples is None
        assert cc.mapie_classifier == None


    def test_fit(self):
        num_examples = 256
        num_classes = 10
        y_hat = torch.rand(num_examples, num_classes).softmax(axis=1).numpy()
        y = y_hat.argmax(axis=1)

        cc = ConformalClassifier()
        cc.append(y_hat, y)
        cc.fit()


    def test_predict(self):
        num_examples = 256
        num_classes = 10
        y_hat = torch.rand(num_examples, num_classes).softmax(axis=1).numpy()
        y = y_hat.argmax(axis=1)

        cc = ConformalClassifier()
        cc.append(y_hat, y)
        cc.fit()

        y_hat = torch.rand(num_examples, num_classes).softmax(axis=1).numpy()
        y = y_hat.argmax(axis=1)

        cc.reset()
        cc.append(y_hat, y)
        results = cc.measure_uncertainty()
        assert len(results["conformal_sets"]) == num_examples
        assert len(results["num_classes"]) == num_examples
        assert len(results["atypical"]) == num_examples
        assert len(results["realized"]) == num_examples
        assert len(results["confused"]) == num_examples
        assert len(results["uncertain"]) == num_examples

    def test_accumulate_predict(self):
        num_examples = 256
        num_classes = 10
        y_hat = torch.rand(num_examples, num_classes).softmax(axis=1).numpy()
        y = y_hat.argmax(axis=1)

        cc = ConformalClassifier()
        cc.append(y_hat, y)
        cc.fit()

        cc.reset()

        y_hat = torch.rand(num_examples, num_classes).softmax(axis=1).numpy()
        y = y_hat.argmax(axis=1)
        cc.append(y_hat, y)

        y_hat = torch.rand(num_examples, num_classes).softmax(axis=1).numpy()
        y = y_hat.argmax(axis=1)
        cc.append(y_hat, y)

        results = cc.measure_uncertainty()
        assert len(results["conformal_sets"]) == 2 * num_examples
        assert len(results["num_classes"]) == 2 * num_examples
        assert len(results["atypical"]) == 2 * num_examples
        assert len(results["realized"]) == 2 * num_examples
        assert len(results["confused"]) == 2 * num_examples
        assert len(results["uncertain"]) == 2 * num_examples
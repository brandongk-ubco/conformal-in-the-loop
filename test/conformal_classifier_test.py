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
        conformal_sets, results = cc.measure_uncertainty()
        assert len(conformal_sets) == num_examples
        assert len(results["prediction_set_size"]) == num_examples
        assert len(results["atypical"]) == num_examples
        assert len(results["realized"]) == num_examples
        assert len(results["confused"]) == num_examples
        assert len(results["uncertain"]) == num_examples

    def test_predict_percentage(self):
        num_examples = 256
        num_classes = 10
        y_hat = torch.rand(num_examples, num_classes).softmax(axis=1).numpy()
        y = y_hat.argmax(axis=1)

        cc = ConformalClassifier()
        cc.append(y_hat, y)
        cc.fit(percentage=0.5)
        conformal_sets, results = cc.measure_uncertainty(percentage=0.5)
        assert len(conformal_sets) == num_examples // 2
        assert len(results["prediction_set_size"]) == num_examples // 2
        assert len(results["atypical"]) == num_examples // 2
        assert len(results["realized"]) == num_examples // 2
        assert len(results["confused"]) == num_examples // 2
        assert len(results["uncertain"]) == num_examples // 2

    def test_predict_uneven_percentage(self):
        num_examples = 256
        num_classes = 10
        y_hat = torch.rand(num_examples, num_classes).softmax(axis=1).numpy()
        y = y_hat.argmax(axis=1)

        cc = ConformalClassifier()
        cc.append(y_hat, y)
        cc.fit(percentage=0.2)
        conformal_sets, results = cc.measure_uncertainty(percentage=0.8)
        assert len(conformal_sets) == int(num_examples * 0.8)
        assert len(results["prediction_set_size"]) == int(num_examples * 0.8)
        assert len(results["atypical"]) == int(num_examples * 0.8)
        assert len(results["realized"]) == int(num_examples * 0.8)
        assert len(results["confused"]) == int(num_examples * 0.8)
        assert len(results["uncertain"]) == int(num_examples * 0.8)


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

        conformal_sets, results = cc.measure_uncertainty()
        assert len(conformal_sets) == 2 * num_examples
        assert len(results["prediction_set_size"]) == 2 * num_examples
        assert len(results["atypical"]) == 2 * num_examples
        assert len(results["realized"]) == 2 * num_examples
        assert len(results["confused"]) == 2 * num_examples
        assert len(results["uncertain"]) == 2 * num_examples
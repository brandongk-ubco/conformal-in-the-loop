from citl.ConformalClassifier import ConformalClassifier
import numpy as np
import torch

class TestConformalClassifier:
    def test_initialize(self):
        cc = ConformalClassifier()
        assert cc.__sklearn_is_fitted__() == True
        assert len(cc.cp_examples) == 0
        assert cc.mapie_classifier == None


    def test_fit(self):
        num_examples = 256
        num_classes = 10
        y_hat = torch.rand(num_examples, num_classes)
        y = y_hat.argmax(axis=1)

        cc = ConformalClassifier()
        cc.append(y_hat, y)
        cc.fit()


    def test_predict(self):
        num_examples = 256
        num_classes = 10
        y_hat = torch.rand(num_examples, num_classes)
        y = y_hat.argmax(axis=1)

        cc = ConformalClassifier()
        cc.append(y_hat, y)
        cc.fit()

        y_hat = torch.rand(num_examples, num_classes)
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

    def test_accumulate_predict(self):
        num_examples = 256
        num_classes = 10
        y_hat = torch.rand(num_examples, num_classes)
        y = y_hat.argmax(axis=1)

        cc = ConformalClassifier()
        cc.append(y_hat, y)
        cc.fit()

        cc.reset()

        y_hat = torch.rand(num_examples, num_classes)
        y = y_hat.argmax(axis=1)
        cc.append(y_hat, y)

        y_hat = torch.rand(num_examples, num_classes)
        y = y_hat.argmax(axis=1)
        cc.append(y_hat, y)

        conformal_sets, results = cc.measure_uncertainty()
        assert len(conformal_sets) == 2 * num_examples
        assert len(results["prediction_set_size"]) == 2 * num_examples
        assert len(results["atypical"]) == 2 * num_examples
        assert len(results["realized"]) == 2 * num_examples
        assert len(results["confused"]) == 2 * num_examples
        assert len(results["uncertain"]) == 2 * num_examples


    def test_fit_two_dimensions(self):
        num_examples = 3
        num_classes = 10
        num_x = 4
        num_y = 5
        y_hat = torch.rand(num_examples, num_classes, num_x, num_y)
        y = y_hat.argmax(axis=1)

        cc = ConformalClassifier()
        cc.append(y_hat, y)
        cc.fit()

    def test_fit_two_dimensions_with_sampling(self):
        num_examples = 3
        num_classes = 10
        num_x = 4
        num_y = 5
        y_hat = torch.rand(num_examples, num_classes, num_x, num_y)
        y = y_hat.argmax(axis=1)

        cc = ConformalClassifier()
        cc.append(y_hat, y, percent=0.5)
        assert(len(cc.val_labels) < num_examples * num_x * num_y)
        cc.fit()

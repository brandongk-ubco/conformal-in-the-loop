from citl.ConformalClassifier import ConformalClassifier
from citl.ConformalClassifier import lac
import torch

class TestConformalClassifier:
    def test_lac(self):
        y_hat = torch.tensor([
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.5],
            ])
        y = torch.tensor([3, 1, 1, 4])
        expected = torch.tensor([ 0.8, 0.0, 1.0, 0.5])
        for i in range(4):
            assert lac(y_hat[i], y[i]) == expected[i]

        
    def test_initialize(self):
        cc = ConformalClassifier()
        assert len(cc.cp_examples) == 0
        assert len(cc.val_labels) == 0

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



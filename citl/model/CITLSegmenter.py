import os
import numpy as np
from mapie.classification import MapieClassifier
import matplotlib.pyplot as plt

path = "E:\BreastRT\Sample Predictions and Masks for Conformal"

mapie_alpha = 0.10
mapie_method = "aps"

predictions = os.listdir(path)

calib_size = len(predictions) // 3
calib_predictions = predictions[:calib_size]
val_predictions = predictions[calib_size:]

class UncertainClassifier():

    def __init__(self, num_classes):
        self.classes_ = range(num_classes)
        self.cp_examples = []

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, x, y):
        raise NotImplementedError("Cannot fit this way.")

    def predict(self, x):
        return self.predict_proba(x).argmax(axis=1)

    def predict_proba(self, x):
        return self.cp_examples[x]

uq_classifier = UncertainClassifier(num_classes=4)

masks = []
predictions = []

for prediction in calib_predictions:
    prediction_dir = os.path.join(path, prediction)
    image_file = os.path.join(prediction_dir, "image.npy")
    mask_file = os.path.join(prediction_dir, "mask.npy")
    prediction_file = os.path.join(prediction_dir, "prediction.npy")
    image = np.load(image_file)
    mask = np.load(mask_file).squeeze()
    prediction = np.load(prediction_file).squeeze()

    mask = mask.flatten()
    prediction = prediction.reshape(-1, prediction.shape[-1])
    masks.append(mask)
    predictions.append(prediction)

masks = np.concatenate(masks)
predictions = np.concatenate(predictions)
uq_classifier.cp_examples = predictions

mapie_classifier = MapieClassifier(
    estimator=uq_classifier, method=mapie_method, cv="prefit", n_jobs=-1
).fit(np.array(range(len(masks))), masks)

for prediction in val_predictions:
    prediction_dir = os.path.join(path, prediction)
    image_file = os.path.join(prediction_dir, "image.npy")
    mask_file = os.path.join(prediction_dir, "mask.npy")
    prediction_file = os.path.join(prediction_dir, "prediction.npy")
    image = np.load(image_file)
    mask = np.load(mask_file).squeeze()
    prediction = np.load(prediction_file).squeeze()

    uq_classifier.cp_examples = prediction.reshape(-1, prediction.shape[-1])

    uq_predictions = mapie_classifier.predict(
                range(len(uq_classifier.cp_examples)), alpha=[mapie_alpha]
            )
    
    num_classes = uq_predictions[1].sum(axis=1).squeeze().reshape(prediction.shape[:-1])

    mask = np.ma.masked_where(mask == 0, mask)
    num_classes = np.ma.masked_where(num_classes == 1, num_classes)
    prediction = prediction.argmax(axis=2)
    prediction = np.ma.masked_where(prediction == 0, prediction)

    image = image / image.max()
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, cmap='jet', interpolation='none', alpha=0.2, vmin=1, vmax=4)
    plt.grid(False)
    plt.axis('off')
    plt.title("Ground Truth")

    plt.subplot(1,2,2)
    plt.imshow(image, cmap='gray')
    plt.imshow(prediction, cmap='jet', interpolation='none', alpha=0.2, vmin=1, vmax=4)
    plt.imshow(num_classes, cmap='Reds', interpolation='none', vmin=1, vmax=4)
    plt.grid(False)
    plt.axis('off')
    plt.title("Prediction")

    plt.show()
    import pdb
    pdb.set_trace()
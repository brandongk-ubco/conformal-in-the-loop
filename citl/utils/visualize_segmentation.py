import numpy as np
from matplotlib import pyplot as plt


def visualize_segmentation(image, mask, prediction=None, uncertainty=None):

    mask = np.ma.masked_where(mask == 0, mask)

    if uncertainty is not None and prediction is None:
        raise ValueError("Cannot visualize uncertainty without a prediction")

    if uncertainty is not None:
        prediction_set_size = uncertainty["prediction_set_size"]
        prediction_set_size = np.ma.masked_where(
            prediction_set_size == 1, prediction_set_size
        )

    num_subplots = 2 if prediction is not None or uncertainty is not None else 1
    num_classes = prediction.shape[0]

    image = image / image.max()
    fig = plt.figure()
    plt.subplot(1, num_subplots, 1)
    plt.imshow(image, cmap="gray")
    plt.imshow(mask, cmap="jet", interpolation="none", alpha=0.2, vmin=1, vmax=4)
    plt.grid(False)
    plt.axis("off")
    plt.title("Ground Truth")

    if num_subplots == 2:
        plt.subplot(1, 2, 2)
        prediction = prediction.argmax(axis=2)
        prediction = np.ma.masked_where(prediction == 0, prediction)
        plt.imshow(image, cmap="gray")
        plt.imshow(
            prediction,
            cmap="jet",
            interpolation="none",
            alpha=0.2,
            vmin=1,
            vmax=num_classes,
        )
        if prediction_set_size is not None:
            plt.imshow(
                prediction_set_size,
                cmap="Reds",
                interpolation="none",
                vmin=1,
                vmax=num_classes,
            )
        plt.grid(False)
        plt.axis("off")
        plt.title("Prediction")

    return fig

import numpy as np
from matplotlib import pyplot as plt


def visualize_segmentation(image, mask=None, prediction=None, prediction_set_size=None):

    num_subplots = 0
    if prediction is not None:
        num_subplots += 1
    if prediction_set_size is not None:
        num_subplots += 1
    if mask is not None:
        num_subplots += 1

    if num_subplots == 0:
        raise ValueError(
            "At least one of mask, prediction or prediction_set_size must be provided."
        )

    num_classes = None

    image = image / image.max()
    image = image - image.min()

    subplot = 1

    fig = plt.figure()

    if image.shape[0] > image.shape[1]:
        mode = "colwise"
    else:
        mode = "rowwise"

    if mask is not None:
        mask = np.ma.masked_where(mask == 0, mask)

        if mode == "colwise":
            plt.subplot(1, num_subplots, subplot)
        else:
            plt.subplot(num_subplots, 1, subplot)

        if image.ndim == 2 or image.shape[-1] == 1:
            plt.imshow(image, cmap="gray")
        elif image.shape[-1] == 3:
            plt.imshow(image)
        else:
            raise ValueError(f"Image has invalid shape: {image.shape}")

        plt.imshow(mask, cmap="jet", interpolation="none", alpha=0.2, vmin=1, vmax=num_classes)
        plt.grid(False)
        plt.axis("off")
        plt.title("Ground Truth")

        subplot += 1

    if prediction is not None:
        num_classes = prediction.shape[0]

        if mode == "colwise":
            plt.subplot(1, num_subplots, subplot)
        else:
            plt.subplot(num_subplots, 1, subplot)

        prediction = prediction.argmax(axis=0)
        prediction = np.ma.masked_where(prediction == 0, prediction)

        if image.ndim == 2 or image.shape[-1] == 1:
            plt.imshow(image, cmap="gray")
        elif image.shape[-1] == 3:
            plt.imshow(image)
        else:
            raise ValueError(f"Image has invalid shape: {image.shape}")

        plt.imshow(
            prediction,
            cmap="jet",
            interpolation="none",
            alpha=0.2,
            vmin=1,
            vmax=num_classes,
        )
        plt.grid(False)
        plt.axis("off")
        plt.title("Prediction")

        subplot += 1

    if prediction_set_size is not None:

        prediction_set_size = np.ma.masked_where(
            prediction_set_size == 1, prediction_set_size
        )

        num_classes = prediction_set_size.max() if num_classes is None else num_classes
        if mode == "colwise":
            plt.subplot(1, num_subplots, subplot)
        else:
            plt.subplot(num_subplots, 1, subplot)

        if image.ndim == 2 or image.shape[-1] == 1:
            plt.imshow(image, cmap="gray")
        elif image.shape[-1] == 3:
            plt.imshow(image)
        else:
            raise ValueError(f"Image has invalid shape: {image.shape}")

        plt.imshow(
            prediction_set_size,
            cmap="Reds",
            interpolation="none",
            alpha=0.5,
            vmin=1,
            vmax=num_classes,
        )
        plt.grid(False)
        plt.axis("off")
        plt.title("Uncertainty")

    plt.tight_layout(pad=2.0)
    return fig

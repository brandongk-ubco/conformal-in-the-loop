import os

import cv2
import pytorch_lightning as L
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from citl import cli

from .dataset import Dataset


@cli.command()
def visualize(
    dataset: str,
    split: str = "train",
    augmentation_policy_path: str = "./policies/noop.yaml",
    rows: int | None = None,
    columns: int | None = None,
    min_img_width: int = 128,
    min_img_height: int = 128,
    examples: int = 1,
):
    L.seed_everything(42, workers=True)

    policy, _ = os.path.splitext(os.path.basename(augmentation_policy_path))

    save_dir = os.path.join("visualizations", dataset, split, policy)

    writer = SummaryWriter(save_dir)

    datamodule = Dataset.get(dataset)(augmentation_policy_path)

    if split == "train":
        datamodule.setup()
        dataloader = datamodule.train_dataloader()
    elif split == "val":
        datamodule.setup()
        dataloader = datamodule.val_dataloader()
    elif split == "test":
        datamodule.setup()
        dataloader = datamodule.test_dataloader()
    else:
        raise ValueError(f"Invalid split: {split}")

    classes = datamodule.classes

    dataiter = iter(dataloader)
    max_width = 1920
    max_height = 1080
    img, _ = next(dataiter)

    img_width = max(img.shape[-1], min_img_width)
    img_height = max(img.shape[-2], min_img_height)

    if rows is None:
        rows = max(max_height // img_height, 1)

    if columns is None:
        columns = max(max_width // img_width, 1)

    shown = 0
    for _, batch in enumerate(dataiter):
        batch_size = batch[0].shape[0]
        for i in range(batch_size):
            x, y = batch[0][i, :, :, :], batch[1][i]
            x = x - x.min()
            x = x / x.max()
            label = classes[y]
            writer.add_image(f"{label}", x, shown)
            shown += 1

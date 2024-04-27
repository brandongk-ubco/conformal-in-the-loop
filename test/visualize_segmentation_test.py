import numpy as np
import torch
import os 
from matplotlib import pyplot as plt

from citl.utils.visualize_segmentation import visualize_segmentation

CURRENT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

class TestVisualizeSegmentation:

    def test_visualize_all(self):
        img = torch.load(os.path.join(CURRENT_DIRECTORY, "fixtures", "image.pt"))
        pred = torch.load(os.path.join(CURRENT_DIRECTORY, "fixtures", "prediction.pt"))
        target = torch.load(os.path.join(CURRENT_DIRECTORY, "fixtures", "target.pt"))
        pred_set_size = torch.load(os.path.join(CURRENT_DIRECTORY, "fixtures", "prediction_set_size.pt"))

        visualize_segmentation(img, mask=target, prediction=pred, prediction_set_size=pred_set_size)
        plt.savefig(os.path.join(CURRENT_DIRECTORY, "test_visualize_all.png"))

    def test_visualize_ground_truth(self):
        img = torch.load(os.path.join(CURRENT_DIRECTORY, "fixtures", "image.pt"))
        target = torch.load(os.path.join(CURRENT_DIRECTORY, "fixtures", "target.pt"))

        visualize_segmentation(img, mask=target)
        plt.savefig(os.path.join(CURRENT_DIRECTORY, "test_visualize_target.png"))

    def test_visualize_prediction(self):
        img = torch.load(os.path.join(CURRENT_DIRECTORY, "fixtures", "image.pt"))
        pred = torch.load(os.path.join(CURRENT_DIRECTORY, "fixtures", "prediction.pt"))

        visualize_segmentation(img, prediction=pred)
        plt.savefig(os.path.join(CURRENT_DIRECTORY, "test_visualize_prediction.png"))

    def test_visualize_uncertainty(self):
        img = torch.load(os.path.join(CURRENT_DIRECTORY, "fixtures", "image.pt"))
        pred_set_size = torch.load(os.path.join(CURRENT_DIRECTORY, "fixtures", "prediction_set_size.pt"))

        visualize_segmentation(img, prediction_set_size=pred_set_size)
        plt.savefig(os.path.join(CURRENT_DIRECTORY, "test_visualize_uncertainty.png"))
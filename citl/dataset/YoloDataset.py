import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class YoloDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        self.labels = np.load(os.path.join(labels_dir, 'labels.npz'), allow_pickle=True)["labels"]
        self.images = np.load(os.path.join(images_dir, 'images.npz'), allow_pickle=True)["images"]
        assert len(self.image_files) == len(self.labels)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        image = self.images[idx]

        boxes = []
        labels = []

        instance_labels = self.labels[idx]

        for label in instance_labels:
            class_id = int(label[0])
            x_center = float(label[1])
            y_center = float(label[2])
            width = float(label[3])
            height = float(label[4])
            # Convert from YOLO format to [x_min, y_min, x_max, y_max]
            x_min = (x_center - width / 2) * image.shape[1]
            y_min = (y_center - height / 2) * image.shape[0]
            x_max = (x_center + width / 2) * image.shape[1]
            y_max = (y_center + height / 2) * image.shape[0]
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels
        }

        return image, target
    
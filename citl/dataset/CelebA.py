import os
from os.path import join
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset
import PIL
import pandas as pd
import numpy as np
import zipfile
from functools import partial
from torchvision.datasets.utils import download_file_from_google_drive, check_integrity, verify_str_arg
import torchvision.transforms as transforms
import pytorch_lightning as L

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")


class CelebA(VisionDataset):
    base_folder = "celeba"

    def __init__(self, root, split="train", target_type="attr", transform=None, target_transform=None, target_attr="Attractive", labelwise=False):
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self.split = split
        self.target_type = [target_type] if isinstance(target_type, str) else target_type
        self.sensitive_attr = 'Male'
        self.target_attr = target_attr
        self.labelwise = labelwise

        if not self.target_type and self.target_transform:
            raise RuntimeError("target_transform is specified but target_type is empty")

        # Handling split and reading files
        split_map = {"train": 0, "valid": 1, "test": 2, "all": None}
        split = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]

        fn = partial(join, self.root, self.base_folder)
        splits = pd.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pd.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None) if split is None else (splits[1] == split)
        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # Map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

        self.target_idx = self.attr_names.index(self.target_attr)
        self.sensi_idx = self.attr_names.index(self.sensitive_attr)
        self.feature_idx = [i for i in range(len(self.attr_names)) if i != self.target_idx and i != self.sensi_idx]

        self.num_classes = 2
        self.num_groups = 2
        self.num_data = self._data_count()

        if self.split == "test":
            self._balance_test_data()

        if self.labelwise:
            self.idx_map = self._make_idx_map()

    def _make_idx_map(self):
        idx_map = [[] for _ in range(self.num_groups * self.num_classes)]
        for j, i in enumerate(self.attr):
            y = self.attr[j, self.target_idx]
            s = self.attr[j, self.sensi_idx]
            pos = s * self.num_classes + y
            idx_map[pos].append(j)
        return [item for sublist in idx_map for item in sublist]

    def __getitem__(self, index):
        if self.labelwise:
            index = self.idx_map[index]
        img_name = self.filename[index]
        X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", img_name))

        target = self.attr[index, self.target_idx]
        sensitive = self.attr[index, self.sensi_idx]
        feature = self.attr[index, self.feature_idx]

        if self.transform:
            X = self.transform(X)

        if self.target_transform:
            target = self.target_transform(target)

        return X, target, sensitive

    def __len__(self):
        return len(self.attr)

    def _data_count(self):
        data_count = np.zeros((self.num_groups, self.num_classes), dtype=int)
        for index in range(len(self.attr)):
            target = self.attr[index, self.target_idx]
            sensitive = self.attr[index, self.sensi_idx]
            data_count[sensitive, target] += 1
        return data_count

    def _balance_test_data(self):
        num_data_min = np.min(self.num_data)
        data_count = np.zeros((self.num_groups, self.num_classes), dtype=int)
        new_filename = []
        new_attr = []

        for index in range(len(self.attr)):
            target = self.attr[index, self.target_idx]
            sensitive = self.attr[index, self.sensi_idx]
            if data_count[sensitive, target] < num_data_min:
                new_filename.append(self.filename[index])
                new_attr.append(self.attr[index])
                data_count[sensitive, target] += 1

        self.filename = new_filename
        self.attr = torch.stack(new_attr)

PATH_DATASETS = os.environ.get("PATH_DATASETS", "./")

class CelebADataModule(L.LightningDataModule):
        
        classes = [
        "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
        "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
        "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
        "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
        "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
        "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
        "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
        "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie",
        "Young"
            ]
        
        task = "classification"
    
    
        def __init__(self, batch_size=128, data_dir=PATH_DATASETS, image_size=224):
            super().__init__()
            self.data_dir = data_dir
            self.batch_size = 128
            self.data_dir = data_dir
            self.image_size = image_size
            self.num_classes = 2
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        

        def prepare_data(self):
            CelebA(self.data_dir, split="train")
            CelebA(self.data_dir, split="valid")
            CelebA(self.data_dir, split="test")

        def setup(self, stage=None):
            if stage == 'fit' or stage is None:
                self.celeba_train = CelebA(self.data_dir, split="train", transform=self.transform)
                self.celeba_val = CelebA(self.data_dir, split="valid", transform=self.transform)

            if stage == 'test' or stage is None:
                self.celeba_test = CelebA(self.data_dir, split="test", transform=self.transform)

        def train_dataloader(self):
            return DataLoader(self.celeba_train, batch_size=self.batch_size, shuffle=True, num_workers=os.cpu_count(), persistent_workers=True)

        def val_dataloader(self):
            return DataLoader(self.celeba_val, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(), persistent_workers=True)

        def test_dataloader(self):
            return DataLoader(self.celeba_test, batch_size=self.batch_size, shuffle=False, num_workers=os.cpu_count(), persistent_workers=True)

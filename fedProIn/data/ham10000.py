import os
import pandas as pd
import torch
from PIL import Image
import torchvision
from torch.utils.data import Dataset
from pathlib import Path


data_dir = Path(__file__).resolve().parent.parent.parent / "dataset" / "HAM10000"

class pytorch_dataset(Dataset):
    def __init__(self, split="train", image_size=(128, 128), as_rgb=True, transform=None, target_transform=None):
        self.split = split
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        elif isinstance(image_size, tuple) and len(image_size) == 1:
            image_size = (image_size[0], image_size[0])
        self.image_size = image_size
        self.as_rgb = as_rgb
        self.train_data = pd.read_csv(os.path.join(data_dir, "HAM10000_metadata_with_images_train.csv"))
        self.val_data = pd.read_csv(os.path.join(data_dir, "HAM10000_metadata_with_images_val.csv"))
        self.test_data = pd.read_csv(os.path.join(data_dir, "HAM10000_metadata_with_images_test.csv"))
        self.labels = self.train_data["dx"].unique().tolist()
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.index_to_label = {idx: label for idx, label in enumerate(self.labels)}
        if self.split == "train":
            self.data = self.train_data
        elif self.split == "val":
            self.data = self.val_data
        elif self.split == "test":
            self.data = self.test_data
        else:
            raise ValueError("split must be one of 'train', 'val', or 'test'")
        
        resize_transform = [torchvision.transforms.Resize(self.image_size)]
        
        default_transform = [
            torchvision.transforms.ToTensor()
        ]
        if self.split != "train":
            normalize_transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            default_transform = default_transform + [normalize_transform]
            
        if transform is not None:
            if isinstance(transform, torchvision.transforms.Compose):
                user_transforms = transform.transforms
            else:
                user_transforms = transform
        else:
            user_transforms = []
        
        all_transforms = resize_transform  + user_transforms + default_transform
        self.transform = torchvision.transforms.Compose(all_transforms)

        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = data_dir / self.data.iloc[idx]["image_path"]
        image = Image.open(image_path)
        if not self.as_rgb:
            image = image.convert("L")
        if self.transform:
            image = self.transform(image)
        label = self.data.iloc[idx]["dx"]
        label = self.label_to_index[label]
        label = torch.tensor(label, dtype=torch.long)
        return image, label
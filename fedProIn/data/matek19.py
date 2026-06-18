import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path

csv_path = Path(__file__).resolve().parent.parent.parent / "dataset" / "MATEK19" / "data" / "Matek-19 Dataset" / "matek19.csv"
data_dir = csv_path.parent

label_to_index = {
    "LYT": 0,
    "NGS": 1,
    "MON": 2,
    "MYO": 3,
    "EOS": 4,
    "MYB": 5,
    "BAS": 6,
    "PMO": 7,
    "NGB": 8,
    "MOB": 9,
    "EBO": 10,
    "LYA": 11,
    "KSC": 12,
    "PMB": 13,
    "MMZ": 14,
}

class pytorch_dataset(Dataset):
    def __init__(self, split="train", image_size=(128, 128), as_rgb=True, transform=None, target_transform=None):
        self.split = split
        self.data = pd.read_csv(csv_path)
        self.data = self.data[self.data["split"] == self.split].reset_index(drop=True)

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        elif isinstance(image_size, tuple) and len(image_size) == 1:
            image_size = (image_size[0], image_size[0])

        self.image_size = image_size
        self.as_rgb = as_rgb

        resize_transform = [transforms.Resize(self.image_size)]
        default_transform = [transforms.ToTensor()]

        if self.split != "train":
            normalize = (
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                if self.as_rgb
                else transforms.Normalize(mean=[0.5], std=[0.5])
            )
            default_transform.append(normalize)

        if transform is not None:
            if isinstance(transform, transforms.Compose):
                user_transforms = transform.transforms
            else:
                user_transforms = transform
        else:
            user_transforms = []

        self.transform = transforms.Compose(resize_transform + user_transforms + default_transform)

        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(data_dir / row["converted_abs_path"])
        image = image.convert("RGB" if self.as_rgb else "L")

        label = torch.tensor(label_to_index[row["label"]], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
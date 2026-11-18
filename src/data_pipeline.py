
from pathlib import Path
import csv
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "oxford-iiit-pet"
SPLITS_DIR = BASE_DIR / "splits"

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0

# Official preprocessing for pretrained weights
WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT

# Same preprocessing, plus train-only augmentation
train_transform = transforms.Compose([
    WEIGHTS.transforms(),
    transforms.RandomHorizontalFlip(p=0.5),
])

# No random augmentation for validation/test
eval_transform = WEIGHTS.transforms()


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PetDataset(Dataset):
    def __init__(self, split_name, transform=None):
        self.transform = transform
        self.records = []

        csv_path = SPLITS_DIR / f"{split_name}.csv"

        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                self.records.append({
                    "filename": row["filename"],
                    "label": int(row["label"]),
                    "breed": row["breed"],
                })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image_path = DATA_DIR / "images" / record["filename"]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = record["label"]

        return image, label


def create_dataloaders(batch_size=BATCH_SIZE):
    train_dataset = PetDataset(
        split_name="train",
        transform=train_transform,
    )

    val_dataset = PetDataset(
        split_name="validation",
        transform=eval_transform,
    )

    test_dataset = PetDataset(
        split_name="test",
        transform=eval_transform,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader

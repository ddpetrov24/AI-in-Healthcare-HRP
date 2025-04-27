import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LABEL_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


class BrainTumor(Dataset):
    #Brain Tumor dataset for classification

    def __init__(
        self,
        dataset_path: str,
        transform_pipeline: str = "default",
    ):
        self.transform = self.get_transform(transform_pipeline)
        self.data = []

        with open(Path(dataset_path, "labels.csv"), newline="") as f:
            for fname, label in csv.reader(f):
                if label in LABEL_NAMES:
                    img_path = Path(dataset_path, fname)
                    label_id = LABEL_NAMES.index(label)

                    self.data.append((img_path, label_id))

    def get_transform(self, transform_pipeline: str = "default"):
        xform = None

        if transform_pipeline == "default":
            xform = transforms.Compose([
                 transforms.Resize((224,224)),
                 transforms.ToTensor()
            ])
        elif transform_pipeline == "aug":
            xform = transforms.Compose(
                [
                    transforms.Resize((224,224)),
                    transforms.RandomHorizontalFlip(p=0.5),       # Flip image with 50% probability
                    transforms.RandomRotation(degrees=10),        # Randomly rotate by ±10 degrees
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, etc.
                    transforms.ToTensor(),
                ]
            )

        if xform is None:
            raise ValueError(f"Invalid transform {transform_pipeline} specified!")

        return xform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label_id = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        data = (self.transform(img), label_id)

        return data


def load_data(
    dataset_path: str,
    transform_pipeline: str = "default",
    return_dataloader: bool = True,
    num_workers: int = 4,
    batch_size: int = 128,
    shuffle: bool = False,
) -> DataLoader | Dataset:
    dataset = BrainTumor(dataset_path, transform_pipeline=transform_pipeline)

    if not return_dataloader:
        return dataset

    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )

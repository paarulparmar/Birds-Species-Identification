"""Dataset helpers for CUB-200-2011 and class-folder image datasets."""
from pathlib import Path
from typing import Optional

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224, train: bool = False):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CUB200Dataset(Dataset):
    """Reads the official CUB-200-2011 annotation files and split."""

    def __init__(self, root: str | Path, train: bool, transform=None):
        self.root = Path(root)
        self.transform = transform
        images_file = self.root / "images.txt"
        labels_file = self.root / "image_class_labels.txt"
        split_file = self.root / "train_test_split.txt"
        for path in (images_file, labels_file, split_file):
            if not path.exists():
                raise FileNotFoundError(f"Missing {path}. Set --data-dir to the CUB-200-2011 directory.")
        images = {int(i): name for i, name in (line.split(maxsplit=1) for line in images_file.read_text().splitlines())}
        labels = {int(i): int(label) - 1 for i, label in (line.split() for line in labels_file.read_text().splitlines())}
        split = {int(i): int(flag) for i, flag in (line.split() for line in split_file.read_text().splitlines())}
        self.samples = [(self.root / "images" / images[i], labels[i]) for i in images if split[i] == int(train)]
        class_file = self.root / "classes.txt"
        self.classes = [line.split(maxsplit=1)[1].replace("_", " ") for line in class_file.read_text().splitlines()] if class_file.exists() else sorted({y for _, y in self.samples})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageFolderDataset(Dataset):
    """Class folders: root/class_name/image.jpg; split should contain train/ and val/."""

    def __init__(self, root: str | Path, transform=None, classes: Optional[list[str]] = None):
        self.root = Path(root)
        self.transform = transform
        if not self.root.is_dir():
            raise FileNotFoundError(f"Image folder not found: {self.root}")
        found = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.classes = classes or found
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.samples = [(p, self.class_to_idx[d.name]) for d in self.root.iterdir() if d.is_dir() and d.name in self.class_to_idx for p in d.rglob("*") if p.suffix.lower() in extensions]
        if not self.samples:
            raise ValueError(f"No supported images found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def make_loaders(data_dir: str | Path, dataset: str = "cub", batch_size: int = 32,
                 workers: int = 4, image_size: int = 224, pin_memory: bool = True):
    root = Path(data_dir)
    if dataset == "cub":
        train_ds = CUB200Dataset(root, True, build_transforms(image_size, True))
        val_ds = CUB200Dataset(root, False, build_transforms(image_size, False))
    else:
        train_ds = ImageFolderDataset(root / "train", build_transforms(image_size, True))
        val_ds = ImageFolderDataset(root / "val", build_transforms(image_size, False), train_ds.classes)
    if train_ds.classes != val_ds.classes:
        raise ValueError("Train and validation class order differs")
    common = dict(batch_size=batch_size, num_workers=workers, pin_memory=pin_memory,
                  persistent_workers=workers > 0)
    return (DataLoader(train_ds, shuffle=True, **common),
            DataLoader(val_ds, shuffle=False, **common), train_ds.classes)

"""Run batched GPU inference on every supported image in a directory."""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset import IMAGENET_MEAN, IMAGENET_STD
from model import build_model


class ImagePaths(Dataset):
    def __init__(self, paths, size):
        self.paths = paths
        self.transform = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(self.paths[i])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best.pth")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=1)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = checkpoint["classes"]
    model = build_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    paths = sorted(p for p in Path(args.input_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
    if not paths:
        raise SystemExit(f"No supported images found in {args.input_dir}")
    size = checkpoint.get("image_size", 224)
    loader = DataLoader(ImagePaths(paths, size), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "prediction", "confidence"])
        with torch.no_grad():
            for images, names in loader:
                logits = model(images.to(device, non_blocking=True))
                probs, indices = logits.softmax(1).topk(min(args.top_k, len(classes)), dim=1)
                for name, row_p, row_i in zip(names, probs.cpu(), indices.cpu()):
                    for probability, idx in zip(row_p.tolist(), row_i.tolist()):
                        writer.writerow([name, classes[idx], f"{probability:.6f}"])
    print(f"Wrote predictions for {len(paths)} images to {args.output} using {device}.")


if __name__ == "__main__":
    main()

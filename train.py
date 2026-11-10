"""Train a ResNet-50 on CUB-200-2011 or train/val class folders."""
import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import make_loaders
from model import build_model


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp=False):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = seen = 0
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
    return total_loss / seen, correct / seen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="CUB root or ImageFolder root with train/ and val/")
    parser.add_argument("--dataset", choices=("cub", "folder"), default="cub")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--unfreeze-from", choices=("layer4", "layer3", "layer2", "layer1", "all"), default="layer4")
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, classes = make_loaders(args.data_dir, args.dataset, args.batch_size, args.workers, args.image_size, device.type == "cuda")
    model = build_model(len(classes), not args.no_pretrained, args.freeze_backbone, args.unfreeze_from).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, amp=amp)
        scheduler.step()
        print(f"Epoch {epoch:03d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.3%} | val loss {val_loss:.4f} acc {val_acc:.3%}")
        state = {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "val_accuracy": val_acc, "classes": classes, "image_size": args.image_size}
        torch.save(state, out / "last.pth")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(state, out / "best.pth")
    print(f"Best validation accuracy: {best_acc:.3%}")


if __name__ == "__main__":
    main()

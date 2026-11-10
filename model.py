"""ResNet-50 classifier initialized from ImageNet weights."""
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


def build_model(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False,
                unfreeze_from: str = "layer4"):
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(in_features, num_classes))
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        if unfreeze_from in {"layer4", "layer3", "layer2", "layer1", "all"}:
            for name in ("layer4", "layer3", "layer2", "layer1"):
                if unfreeze_from == "all" or name >= unfreeze_from:
                    for parameter in getattr(model, name).parameters():
                        parameter.requires_grad = True
    return model

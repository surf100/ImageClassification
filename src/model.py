
import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


NUM_CLASSES = 37


def create_model(num_classes=NUM_CLASSES):
    # Load the pretrained MobileNetV3-Small
    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = mobilenet_v3_small(weights=weights)

    # Find the number of input features
    in_features = model.classifier[3].in_features

    # Replace the final classification layer
    model.classifier[3] = nn.Linear(
        in_features,
        num_classes,
    )

    return model


def count_parameters(model):
    total = sum(
        p.numel() for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable

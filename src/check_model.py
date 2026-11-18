
from pathlib import Path

import torch
import torchvision

from model import create_model, count_parameters


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def main():
    print("=== Step 5: MobileNetV3-Small Check ===")

    print("\n=== Library Versions ===")
    print("PyTorch:", torch.__version__)
    print("TorchVision:", torchvision.__version__)

    # Make random initialization reproducible
    torch.manual_seed(42)

    # Create pretrained model with 37 output classes
    model = create_model(num_classes=37)

    print("\n=== Model Architecture ===")
    print("Model:", type(model).__name__)

    print("\nFeature extractor:")
    print(type(model.features).__name__)

    print("\nPooling:")
    print(model.avgpool)

    print("\nClassifier:")
    print(model.classifier)

    # Parameter counts
    total, trainable = count_parameters(model)

    print("\n=== Parameter Counts ===")
    print("Total parameters:", f"{total:,}")
    print("Trainable parameters:", f"{trainable:,}")

    assert model.classifier[3].out_features == 37
    assert total == trainable

    # Use CPU for this small architecture test
    model = model.to("cpu")
    model.eval()

    # Simulate a batch of 2 RGB images
    dummy_input = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    print("\n=== Forward Pass Test ===")
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Output dtype:", output.dtype)

    assert output.shape == (2, 37)
    assert torch.isfinite(output).all()

    # Save model information
    output_path = RESULTS_DIR / "model_architecture.txt"

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("MobileNetV3-Small Architecture\n")
        file.write(f"PyTorch: {torch.__version__}\n")
        file.write(f"TorchVision: {torchvision.__version__}\n")
        file.write("Weights: IMAGENET1K_V1 (DEFAULT)\n")
        file.write("Number of classes: 37\n")
        file.write(f"Total parameters: {total}\n")
        file.write(f"Trainable parameters: {trainable}\n")
        file.write(f"Classifier:\n{model.classifier}\n")

    print("\nArchitecture saved:", output_path)

    print("\nModel architecture check passed!")


if __name__ == "__main__":
    main()

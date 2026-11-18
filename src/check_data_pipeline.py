
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from data_pipeline import (
    WEIGHTS,
    create_dataloaders,
    set_seed,
)


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)


def save_batch_preview(images, labels):
    # Reverse ImageNet normalization for visualization
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        image = images[i].cpu()

        image = image * std[:, None, None]
        image = image + mean[:, None, None]
        image = image.clamp(0, 1)

        image = image.permute(1, 2, 0).numpy()

        ax.imshow(image)
        ax.set_title(f"Class ID: {labels[i].item()}")
        ax.axis("off")

    plt.tight_layout()

    output_path = RESULTS_DIR / "preprocessed_batch.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("Batch preview saved:", output_path)


def main():
    set_seed(42)

    print("=== Step 4: Data Pipeline Check ===")

    print("\nPretrained weights:", WEIGHTS)
    print("Official preprocessing:", WEIGHTS.transforms())

    train_loader, val_loader, test_loader = create_dataloaders()

    print("\n=== Dataset Sizes ===")
    print("Train:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))
    print("Test:", len(test_loader.dataset))

    print("\n=== Number of Batches ===")
    print("Train:", len(train_loader))
    print("Validation:", len(val_loader))
    print("Test:", len(test_loader))

    # Inspect only the training batch, not the test set
    images, labels = next(iter(train_loader))

    print("\n=== First Training Batch ===")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)
    print("Min value:", images.min().item())
    print("Max value:", images.max().item())

    assert images.shape == (32, 3, 224, 224)
    assert labels.shape == (32,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64
    assert torch.isfinite(images).all()
    assert torch.all((labels >= 0) & (labels < 37))

    # Validation should be deterministic
    val_image_1, _ = val_loader.dataset[0]
    val_image_2, _ = val_loader.dataset[0]

    assert torch.equal(val_image_1, val_image_2)

    save_batch_preview(images, labels)

    print("\nAll data pipeline checks passed!")


if __name__ == "__main__":
    main()

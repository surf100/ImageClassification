
from pathlib import Path
from time import perf_counter
import csv
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import sklearn

from sklearn.metrics import f1_score
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from data_pipeline import create_dataloaders
from model import create_model, count_parameters


# =====================================
# 1. CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "head_training"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 15
PATIENCE = 4
NUM_CLASSES = 37

CHECKPOINT_PATH = CHECKPOINT_DIR / "head_training_best.pth"


# =====================================
# 2. RANDOM SEEDS
# =====================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================
# 3. FREEZE FEATURE EXTRACTOR
# =====================================

def freeze_features(model):
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Both fully connected layers stay trainable
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return model


# =====================================
# 4. ONE EPOCH
# =====================================

def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()

        # Keep frozen BatchNorm statistics unchanged
        model.features.eval()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    description = "Training" if training else "Validation"

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(
            loader, desc=description, leave=False
        ):
            images = images.to(device)
            labels = labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            predictions = outputs.argmax(dim=1)
            n = labels.size(0)

            total_loss += loss.item() * n
            total_correct += (predictions == labels).sum().item()
            total_samples += n

            all_predictions.extend(
                predictions.detach().cpu().tolist()
            )
            all_labels.extend(labels.cpu().tolist())

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "macro_f1": f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
    }


# =====================================
# 5. SAVE RESULTS
# =====================================

def save_history(history):
    with open(
        RESULTS_DIR / "training_history.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "train_macro_f1",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def plot_history(history):
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for metric, ax, title in [
        ("loss", axes[0], "Cross-Entropy Loss"),
        ("accuracy", axes[1], "Accuracy"),
    ]:
        ax.plot(
            epochs,
            [row[f"train_{metric}"] for row in history],
            label="Training",
        )
        ax.plot(
            epochs,
            [row[f"val_{metric}"] for row in history],
            label="Validation",
        )
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150)
    plt.close()


# =====================================
# 6. MAIN
# =====================================

def main():
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    print("=== Classifier-Head Training ===")
    print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE
    )

    # Fresh ImageNet-pretrained model, not Step 6 checkpoint
    model = create_model(num_classes=NUM_CLASSES)
    model = freeze_features(model)
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)

    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    print("\n=== Trainable Layers ===")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(name, tuple(parameter.shape))

    assert all(
        not p.requires_grad for p in model.features.parameters()
    )
    assert all(
        p.requires_grad for p in model.classifier.parameters()
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    no_improvement = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{MAX_EPOCHS}")

        train = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val = run_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Train: loss={train['loss']:.4f}, "
            f"accuracy={train['accuracy']:.4f}, "
            f"F1={train['macro_f1']:.4f}"
        )
        print(
            f"Val:   loss={val['loss']:.4f}, "
            f"accuracy={val['accuracy']:.4f}, "
            f"F1={val['macro_f1']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "train_macro_f1": train["macro_f1"],
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
        })

        save_history(history)
        plot_history(history)

        if val["macro_f1"] > best_val_f1:
            best_val_f1 = val["macro_f1"]
            best_epoch = epoch
            no_improvement = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_macro_f1": best_val_f1,
            }, CHECKPOINT_PATH)

            print("New best checkpoint saved!")
        else:
            no_improvement += 1
            print(
                f"No improvement: {no_improvement}/{PATIENCE}"
            )

        if no_improvement >= PATIENCE:
            print("\nEarly stopping activated.")
            break

    torch.cuda.synchronize()
    total_time = perf_counter() - start_time

    config = {
        "method": "classifier_head_training",
        "weights": "IMAGENET1K_V1 (DEFAULT)",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "epochs_completed": len(history),
        "patience": PATIENCE,
        "loss_function": "CrossEntropyLoss",
        "optimizer": "AdamW",
        "checkpoint_rule": "highest validation macro F1",
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_layers": "features",
        "trainable_layers": ["classifier.0", "classifier.3"],
        "training_time_seconds": total_time,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "sklearn_version": sklearn.__version__,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms_enforced": False,
        "test_set_used": False,
    }

    with open(
        RESULTS_DIR / "training_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    print("\n=== Training Completed ===")
    print("Best epoch:", best_epoch)
    print("Best validation macro F1:", round(best_val_f1, 4))
    print("Total training time (seconds):", round(total_time, 2))
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Results:", RESULTS_DIR)


if __name__ == "__main__":
    main()

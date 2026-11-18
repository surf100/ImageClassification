
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
RESULTS_DIR = BASE_DIR / "results" / "full_finetuning"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 15
PATIENCE = 4
NUM_CLASSES = 37

CHECKPOINT_PATH = CHECKPOINT_DIR / "full_finetuning_best.pth"


# =====================================
# 2. REPRODUCIBILITY
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
# 3. TRAIN ONE EPOCH
# =====================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    progress = tqdm(loader, desc="Training", leave=False)

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        all_predictions.extend(predictions.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return avg_loss, accuracy, macro_f1


# =====================================
# 4. VALIDATION
# =====================================

def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(
            loader, desc="Validation", leave=False
        ):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return avg_loss, accuracy, macro_f1


# =====================================
# 5. SAVE TRAINING HISTORY
# =====================================

def save_history(history):
    path = RESULTS_DIR / "training_history.csv"

    with open(path, "w", newline="", encoding="utf-8") as file:
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


# =====================================
# 6. PLOT TRAINING CURVES
# =====================================

def plot_history(history):
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        epochs,
        [row["train_loss"] for row in history],
        label="Training",
    )
    axes[0].plot(
        epochs,
        [row["val_loss"] for row in history],
        label="Validation",
    )
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs,
        [row["train_accuracy"] for row in history],
        label="Training",
    )
    axes[1].plot(
        epochs,
        [row["val_accuracy"] for row in history],
        label="Validation",
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "training_curves.png",
        dpi=150,
    )
    plt.close()


# =====================================
# 7. MAIN TRAINING
# =====================================

def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=== Full Fine-Tuning ===")
    print("Device:", device)

    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is not available. Configure GPU first."
        )

    print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=BATCH_SIZE
    )

    model = create_model(num_classes=NUM_CLASSES)

    # Full fine-tuning: all parameters are trainable
    for parameter in model.parameters():
        parameter.requires_grad = True

    model = model.to(device)

    total_params, trainable_params = count_parameters(model)

    print("Total parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = []

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    start_time = perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{MAX_EPOCHS}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )

        print(
            f"Train: loss={train_loss:.4f}, "
            f"accuracy={train_acc:.4f}, F1={train_f1:.4f}"
        )
        print(
            f"Val:   loss={val_loss:.4f}, "
            f"accuracy={val_acc:.4f}, F1={val_f1:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        })

        # Keep results after every completed epoch
        save_history(history)
        plot_history(history)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                CHECKPOINT_PATH,
            )

            print("New best checkpoint saved!")

        else:
            epochs_without_improvement += 1
            print(
                "No improvement:",
                epochs_without_improvement,
                "/", PATIENCE,
            )

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping activated.")
            break

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = perf_counter() - start_time

    config = {
        "method": "full_finetuning",
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
        "training_time_seconds": total_time,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "sklearn_version": sklearn.__version__,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms_enforced": False,
        "checkpoint_path": CHECKPOINT_PATH.relative_to(BASE_DIR).as_posix(),
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

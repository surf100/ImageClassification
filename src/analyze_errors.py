
from pathlib import Path
import csv
import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline import PetDataset, eval_transform
from model import create_model


# =====================================
# 1. CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
TEST_DIR = RESULTS_DIR / "test_evaluation"
OUTPUT_DIR = RESULTS_DIR / "error_analysis"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
NUM_CLASSES = 37
BATCH_SIZE = 32

METHODS = [
    "full_finetuning",
    "head_training",
    "linear_svm",
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================
# 2. HELPERS
# =====================================

def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_csv(path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def get_class_names():
    data = read_json(RESULTS_DIR / "class_names.json")

    return [
        data[str(i)].replace("_", " ")
        for i in range(NUM_CLASSES)
    ]


# =====================================
# 3. ANALYZE SAVED TEST ERRORS
# =====================================

def analyze_method(method, class_names):
    predictions = read_csv(
        TEST_DIR / f"{method}_predictions.csv"
    )

    assert len(predictions) == 3669
    assert [
        int(row["test_index"]) for row in predictions
    ] == list(range(3669))

    y_true = np.array([
        int(row["true_label"]) for row in predictions
    ])

    y_pred = np.array([
        int(row["predicted_label"]) for row in predictions
    ])

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    class_results = []

    for i, breed in enumerate(class_names):
        total = int(cm[i].sum())
        correct = int(cm[i, i])
        errors = total - correct

        recall = correct / total if total else 0.0

        class_results.append({
            "method": method,
            "breed": breed,
            "total_images": total,
            "correct": correct,
            "errors": errors,
            "recall": round(recall, 4),
        })

    # Every directional mistake:
    # true breed -> predicted breed
    confusion_pairs = []

    for true_id in range(NUM_CLASSES):
        for predicted_id in range(NUM_CLASSES):
            if true_id == predicted_id:
                continue

            count = int(cm[true_id, predicted_id])

            if count > 0:
                confusion_pairs.append({
                    "method": method,
                    "true_breed": class_names[true_id],
                    "predicted_breed": class_names[predicted_id],
                    "count": count,
                })

    confusion_pairs.sort(
        key=lambda row: -row["count"]
    )

    class_results.sort(
        key=lambda row: (-row["errors"], row["breed"])
    )

    print(f"\n=== {method} ===")
    print("Total errors:", int((y_true != y_pred).sum()))

    print("\nTop 5 breeds by error count:")

    for row in class_results[:5]:
        print(
            f"{row['breed']:28s} "
            f"errors={row['errors']:3d} "
            f"recall={row['recall']:.4f}"
        )

    print("\nTop 5 directional confusions:")

    for row in confusion_pairs[:5]:
        print(
            f"{row['true_breed']} -> "
            f"{row['predicted_breed']}: {row['count']}"
        )

    save_csv(
        OUTPUT_DIR / f"{method}_class_errors.csv",
        class_results,
        [
            "method", "breed", "total_images",
            "correct", "errors", "recall",
        ],
    )

    save_csv(
        OUTPUT_DIR / f"{method}_confusion_pairs.csv",
        confusion_pairs,
        [
            "method", "true_breed",
            "predicted_breed", "count",
        ],
    )

    return class_results


# =====================================
# 4. EVALUATE SAVED CHECKPOINT ON TRAIN
# =====================================

def evaluate_training_checkpoint(checkpoint_name):
    dataset = PetDataset(
        split_name="train",
        transform=eval_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = create_model(num_classes=NUM_CLASSES)

    checkpoint = torch.load(
        CHECKPOINT_DIR / checkpoint_name,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_true = []
    all_pred = []

    with torch.inference_mode():
        for images, labels in tqdm(
            loader, desc="Training-set evaluation"
        ):
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_true.extend(labels.numpy().tolist())
            all_pred.extend(
                predictions.cpu().numpy().tolist()
            )

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "accuracy": float(
            accuracy_score(all_true, all_pred)
        ),
        "macro_f1": float(
            f1_score(
                all_true, all_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "selected_epoch": int(checkpoint["epoch"]),
    }


# =====================================
# 5. TRAIN VS TEST COMPARISON
# =====================================

def compare_generalization():
    saved_test = read_csv(
        TEST_DIR / "test_metrics.csv"
    )

    test_metrics = {
        row["method"]: row for row in saved_test
    }

    train_metrics = {}

    for method, checkpoint in [
        ("full_finetuning", "full_finetuning_best.pth"),
        ("head_training", "head_training_best.pth"),
    ]:
        print(f"\nEvaluating training set: {method}")

        train_metrics[method] = (
            evaluate_training_checkpoint(checkpoint)
        )

    svm_config = read_json(
        RESULTS_DIR
        / "feature_extraction"
        / "training_config.json"
    )

    train_metrics["linear_svm"] = {
        "accuracy": float(svm_config["train_accuracy"]),
        "macro_f1": float(svm_config["train_macro_f1"]),
        "selected_epoch": None,
    }

    comparison = []

    for method in METHODS:
        train_accuracy = train_metrics[method]["accuracy"]
        train_f1 = train_metrics[method]["macro_f1"]

        test_accuracy = float(
            test_metrics[method]["test_accuracy"]
        )
        test_f1 = float(
            test_metrics[method]["test_macro_f1"]
        )

        row = {
            "method": method,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "accuracy_gap": train_accuracy - test_accuracy,
            "train_macro_f1": train_f1,
            "test_macro_f1": test_f1,
            "macro_f1_gap": train_f1 - test_f1,
            "selected_epoch": train_metrics[method]["selected_epoch"],
        }

        comparison.append(row)

        print(f"\n{method}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy:  {test_accuracy:.4f}")
        print(
            f"Accuracy gap: {(train_accuracy - test_accuracy) * 100:.2f}"
            " percentage points"
        )
        print(f"Train Macro F1: {train_f1:.4f}")
        print(f"Test Macro F1:  {test_f1:.4f}")

    save_csv(
        OUTPUT_DIR / "generalization_comparison.csv",
        comparison,
        [
            "method", "train_accuracy",
            "test_accuracy", "accuracy_gap",
            "train_macro_f1", "test_macro_f1",
            "macro_f1_gap", "selected_epoch",
        ],
    )

    return comparison


# =====================================
# 6. VISUALIZATIONS
# =====================================

def plot_generalization(rows):
    names = [
        "Full Fine-Tuning",
        "Head Training",
        "Linear SVM",
    ]

    x = np.arange(len(names))
    width = 0.35

    train = [r["train_accuracy"] * 100 for r in rows]
    test = [r["test_accuracy"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(x - width / 2, train, width, label="Train")
    ax.bar(x + width / 2, test, width, label="Test")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training vs Test Performance")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "generalization_gap.png",
        dpi=180,
    )
    plt.close()


def plot_difficult_breeds(class_results):
    # Show classes with the lowest recall for Head Training
    weakest = sorted(
        class_results,
        key=lambda row: (row["recall"], row["breed"])
    )[:10]

    names = [row["breed"] for row in weakest]
    recall = [row["recall"] * 100 for row in weakest]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(names, recall)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Recall (%)")
    ax.set_title(
        "10 Lowest-Recall Breeds — Classifier-Head Training"
    )
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "difficult_breeds.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


# =====================================
# 7. MAIN
# =====================================

def main():
    torch.manual_seed(SEED)

    print("=== Step 11: Error Analysis ===")
    print("Device:", device)

    class_names = get_class_names()

    all_class_results = {}

    for method in METHODS:
        all_class_results[method] = analyze_method(
            method, class_names
        )

    comparison = compare_generalization()

    plot_generalization(comparison)
    plot_difficult_breeds(
        all_class_results["head_training"]
    )

    print("\n=== Files Created ===")

    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path.name)

    print("\nError analysis completed successfully!")


if __name__ == "__main__":
    main()

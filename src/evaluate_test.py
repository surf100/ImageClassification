
from pathlib import Path
import csv
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline import PetDataset, eval_transform
from model import create_model
from train_svm import MobileNetFeatureExtractor


# =====================================
# 1. CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results" / "test_evaluation"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
NUM_CLASSES = 37
SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =====================================
# 2. LOAD CLASS NAMES
# =====================================

def load_class_names():
    path = BASE_DIR / "results" / "class_names.json"

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    class_names = [
        data[str(i)].replace("_", " ")
        for i in range(NUM_CLASSES)
    ]

    return class_names


# =====================================
# 3. CREATE TEST LOADER
# =====================================

def create_test_loader():
    dataset = PetDataset(
        split_name="test",
        transform=eval_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    assert len(dataset) == 3669

    return loader


# =====================================
# 4. LOAD NEURAL NETWORK CHECKPOINT
# =====================================

def load_neural_model(checkpoint_name):
    model = create_model(num_classes=NUM_CLASSES)

    checkpoint = torch.load(
        CHECKPOINT_DIR / checkpoint_name,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()

    print("Loaded checkpoint:", checkpoint_name)
    print("Selected epoch:", checkpoint["epoch"])

    return model


# =====================================
# 5. EVALUATE PYTORCH MODEL
# =====================================

def predict_neural_network(model, loader):
    all_predictions = []
    all_labels = []

    model.eval()

    with torch.inference_mode():
        for images, labels in tqdm(
            loader, desc="Neural network test"
        ):
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )
            all_labels.extend(labels.numpy().tolist())

    return (
        np.array(all_labels),
        np.array(all_predictions),
    )


# =====================================
# 6. EVALUATE LINEAR SVM
# =====================================

def predict_svm(loader):
    print("Loading pretrained feature extractor...")

    base_model = create_model(num_classes=NUM_CLASSES)

    extractor = MobileNetFeatureExtractor(base_model)

    for parameter in extractor.parameters():
        parameter.requires_grad = False

    extractor = extractor.to(device)
    extractor.eval()

    classifier = joblib.load(
        CHECKPOINT_DIR / "linear_svm.joblib"
    )

    all_features = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in tqdm(
            loader, desc="SVM feature extraction"
        ):
            images = images.to(device)

            features = extractor(images)
            all_features.append(
                features.cpu().numpy()
            )

            all_labels.extend(labels.numpy().tolist())

    X_test = np.concatenate(all_features, axis=0)
    y_true = np.array(all_labels)

    assert X_test.shape == (3669, 576)

    predictions = classifier.predict(X_test)

    return y_true, predictions


# =====================================
# 7. SAVE CONFUSION MATRIX
# =====================================

def save_confusion_matrix(cm, class_names, method):
    fig, ax = plt.subplots(figsize=(19, 17))

    image = ax.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues",
    )

    fig.colorbar(image, ax=ax, fraction=0.035)

    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))

    ax.set_xticklabels(
        class_names,
        rotation=90,
        fontsize=8,
    )

    ax.set_yticklabels(
        class_names,
        fontsize=8,
    )

    ax.set_xlabel("Predicted Breed")
    ax.set_ylabel("True Breed")
    ax.set_title(f"Confusion Matrix — {method}")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / f"{method}_confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


# =====================================
# 8. EVALUATE AND SAVE METRICS
# =====================================

def evaluate_method(method, y_true, y_pred, class_names):
    assert len(y_true) == 3669
    assert len(y_pred) == 3669

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    save_confusion_matrix(cm, class_names, method)

    np.savetxt(
        RESULTS_DIR / f"{method}_confusion_matrix.csv",
        cm,
        fmt="%d",
        delimiter=",",
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    with open(
        RESULTS_DIR / f"{method}_classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(report, file, indent=2)

    # Same order as test.csv (shuffle=False)
    with open(
        RESULTS_DIR / f"{method}_predictions.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "test_index",
            "true_label",
            "predicted_label",
            "true_breed",
            "predicted_breed",
            "correct",
        ])

        for i, (true, pred) in enumerate(zip(y_true, y_pred)):
            writer.writerow([
                i,
                int(true),
                int(pred),
                class_names[int(true)],
                class_names[int(pred)],
                int(true == pred),
            ])

    print(f"\n=== {method} ===")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")

    return {
        "method": method,
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "test_images": len(y_true),
    }


# =====================================
# 9. MAIN
# =====================================

def main():
    torch.manual_seed(SEED)

    print("=== FINAL TEST EVALUATION ===")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    class_names = load_class_names()
    test_loader = create_test_loader()

    print("Classes:", len(class_names))
    print("Test images:", len(test_loader.dataset))

    results = []

    # Approach 1: Full Fine-Tuning
    print("\nEvaluating Full Fine-Tuning...")

    model = load_neural_model(
        "full_finetuning_best.pth"
    )

    y_true, y_pred = predict_neural_network(
        model, test_loader
    )

    results.append(
        evaluate_method(
            "full_finetuning",
            y_true,
            y_pred,
            class_names,
        )
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Approach 2: Classifier-Head Training
    print("\nEvaluating Classifier-Head Training...")

    model = load_neural_model(
        "head_training_best.pth"
    )

    y_true, y_pred = predict_neural_network(
        model, test_loader
    )

    results.append(
        evaluate_method(
            "head_training",
            y_true,
            y_pred,
            class_names,
        )
    )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Approach 3: Feature Extraction + Linear SVM
    print("\nEvaluating Feature Extraction + Linear SVM...")

    y_true, y_pred = predict_svm(test_loader)

    results.append(
        evaluate_method(
            "linear_svm",
            y_true,
            y_pred,
            class_names,
        )
    )

    # Save final comparison
    output_path = RESULTS_DIR / "test_metrics.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "test_accuracy",
                "test_macro_f1",
                "test_images",
            ],
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n=== FINAL COMPARISON ===")

    for result in results:
        print(
            f"{result['method']:20s} | "
            f"Accuracy: {result['test_accuracy']:.4f} | "
            f"Macro F1: {result['test_macro_f1']:.4f}"
        )

    print("\nResults saved to:", RESULTS_DIR)
    print("Final test evaluation completed successfully!")


if __name__ == "__main__":
    main()

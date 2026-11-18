
from pathlib import Path
from time import perf_counter
import json
import random

import joblib
import numpy as np
import torch
import torchvision
import sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_pipeline import PetDataset, eval_transform
from model import create_model


# =====================================
# 1. CONFIGURATION
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results" / "feature_extraction"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_CLASSES = 37

C_VALUES = [0.01, 0.1, 1.0, 10.0]

MODEL_PATH = CHECKPOINT_DIR / "linear_svm.joblib"


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
# 3. CREATE DATA LOADERS
# =====================================

def create_feature_loaders():
    train_dataset = PetDataset(
        split_name="train",
        transform=eval_transform,
    )

    val_dataset = PetDataset(
        split_name="validation",
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


# =====================================
# 4. FEATURE EXTRACTOR
# =====================================

class MobileNetFeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.features = model.features
        self.avgpool = model.avgpool

    def forward(self, images):
        x = self.features(images)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


# =====================================
# 5. EXTRACT FEATURES
# =====================================

def extract_features(model, loader, device, name):
    model.eval()

    all_features = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in tqdm(
            loader, desc=f"Extracting {name}"
        ):
            images = images.to(
                device, non_blocking=True
            )

            features = model(images)

            all_features.append(
                features.cpu().numpy()
            )
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f"{name} features shape:", X.shape)
    print(f"{name} labels shape:", y.shape)

    assert X.ndim == 2
    assert X.shape[1] == 576
    assert np.isfinite(X).all()

    return X, y


# =====================================
# 6. TRAIN AND SELECT LINEAR SVM
# =====================================

def select_classifier(X_train, y_train, X_val, y_val):
    results = []

    best_model = None
    best_c = None
    best_f1 = -1.0
    best_fit_time = None

    for c_value in C_VALUES:
        print(f"\nTraining Linear SVM: C={c_value}")

        classifier = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(
                C=c_value,
                max_iter=10000,
                random_state=SEED,
            )),
        ])

        start = perf_counter()
        classifier.fit(X_train, y_train)
        fit_time = perf_counter() - start

        predictions = classifier.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        print(f"Validation accuracy: {accuracy:.4f}")
        print(f"Validation macro F1: {macro_f1:.4f}")
        print(f"Classifier training time: {fit_time:.2f}s")

        results.append({
            "C": c_value,
            "validation_accuracy": float(accuracy),
            "validation_macro_f1": float(macro_f1),
            "classifier_training_seconds": fit_time,
        })

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_c = c_value
            best_model = classifier
            best_fit_time = fit_time

    return best_model, best_c, best_f1, best_fit_time, results


# =====================================
# 7. MAIN
# =====================================

def main():
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    print("=== Feature Extraction + Linear SVM ===")
    print("GPU:", torch.cuda.get_device_name(0))

    # Fresh ImageNet-pretrained network
    base_model = create_model(num_classes=NUM_CLASSES)

    # Only features and pooling are used
    feature_extractor = MobileNetFeatureExtractor(
        base_model
    )

    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False

    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    trainable_params = sum(
        p.numel()
        for p in feature_extractor.parameters()
        if p.requires_grad
    )

    assert trainable_params == 0

    print("Trainable neural network parameters:", trainable_params)

    train_loader, val_loader = create_feature_loaders()

    # Warm-up avoids including initial GPU startup in timing
    with torch.inference_mode():
        dummy = torch.zeros(
            1, 3, 224, 224, device=device
        )
        sample_features = feature_extractor(dummy)

    assert sample_features.shape == (1, 576)

    torch.cuda.synchronize()
    start_extraction = perf_counter()

    X_train, y_train = extract_features(
        feature_extractor, train_loader, device, "train"
    )

    X_val, y_val = extract_features(
        feature_extractor, val_loader, device, "validation"
    )

    torch.cuda.synchronize()
    extraction_time = perf_counter() - start_extraction

    assert X_train.shape == (2944, 576)
    assert X_val.shape == (736, 576)
    assert len(np.unique(y_train)) == 37
    assert len(np.unique(y_val)) == 37

    print("\nFeature extraction time:", round(extraction_time, 2), "seconds")

    # Train multiple candidates using training data only
    (
        best_model,
        best_c,
        best_val_f1,
        best_fit_time,
        search_results,
    ) = select_classifier(X_train, y_train, X_val, y_val)

    # Validation-only comparison: no test evaluation
    train_predictions = best_model.predict(X_train)
    val_predictions = best_model.predict(X_val)

    train_accuracy = accuracy_score(y_train, train_predictions)
    train_f1 = f1_score(
        y_train, train_predictions,
        average="macro", zero_division=0
    )

    val_accuracy = accuracy_score(y_val, val_predictions)

    # Save selected sklearn pipeline (scaler + SVM)
    joblib.dump(best_model, MODEL_PATH)

    np.savez_compressed(
        RESULTS_DIR / "train_features.npz",
        features=X_train,
        labels=y_train,
    )

    np.savez_compressed(
        RESULTS_DIR / "validation_features.npz",
        features=X_val,
        labels=y_val,
    )

    config = {
        "method": "feature_extraction_linear_svm",
        "backbone": "MobileNetV3-Small",
        "weights": "IMAGENET1K_V1 (DEFAULT)",
        "feature_location": "after avgpool, before classifier",
        "feature_dimension": 576,
        "neural_network_trainable_parameters": 0,
        "classifier": "LinearSVC",
        "scaling": "StandardScaler fitted on train only",
        "train_augmentation": "none (deterministic features)",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "c_values": C_VALUES,
        "best_C": best_c,
        "checkpoint_rule": "highest validation macro F1",
        "train_accuracy": float(train_accuracy),
        "train_macro_f1": float(train_f1),
        "validation_accuracy": float(val_accuracy),
        "validation_macro_f1": float(best_val_f1),
        "feature_extraction_seconds": extraction_time,
        "selected_classifier_fit_seconds": best_fit_time,
        "total_selected_pipeline_seconds": (
            extraction_time + best_fit_time
        ),
        "hyperparameter_search_results": search_results,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "sklearn_version": sklearn.__version__,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "deterministic_algorithms_enforced": False,
        "test_set_used": False,
        "model_path": str(MODEL_PATH),
    }

    with open(
        RESULTS_DIR / "training_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    print("\n=== Final Results ===")
    print("Best C:", best_c)
    print("Training accuracy:", round(train_accuracy, 4))
    print("Training macro F1:", round(train_f1, 4))
    print("Validation accuracy:", round(val_accuracy, 4))
    print("Validation macro F1:", round(best_val_f1, 4))
    print("Feature extraction time:", round(extraction_time, 2))
    print("Selected SVM fit time:", round(best_fit_time, 2))
    print(
        "Feature extraction + selected SVM time:",
        round(extraction_time + best_fit_time, 2),
    )
    print("Saved classifier:", MODEL_PATH)
    print("\nStep 8 completed successfully!")


if __name__ == "__main__":
    main()

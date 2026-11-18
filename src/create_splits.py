
from pathlib import Path
import csv
import json
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "oxford-iiit-pet"
RESULTS_DIR = BASE_DIR / "results"
SPLITS_DIR = BASE_DIR / "splits"

SEED = 42
VALIDATION_SIZE = 0.20

SPLITS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_annotations(split):
    annotation_file = DATA_DIR / "annotations" / f"{split}.txt"

    records = []

    with open(annotation_file, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            image_id = parts[0]
            label = int(parts[1]) - 1
            breed = image_id.rsplit("_", 1)[0]

            records.append({
                "image_id": image_id,
                "filename": image_id + ".jpg",
                "label": label,
                "breed": breed,
            })

    return records


def save_split(records, filename):
    output_path = SPLITS_DIR / filename

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_id", "filename", "label", "breed"],
        )

        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {output_path.name}: {len(records)} images")


def save_class_distribution(train, val, test):
    class_counts = {}

    for split_name, records in [
        ("train", train),
        ("validation", val),
        ("test", test),
    ]:
        for record in records:
            breed = record["breed"]

            if breed not in class_counts:
                class_counts[breed] = {
                    "train": 0,
                    "validation": 0,
                    "test": 0,
                }

            class_counts[breed][split_name] += 1

    output_path = RESULTS_DIR / "split_class_distribution.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            "breed",
            "train",
            "validation",
            "test",
        ])

        for breed in sorted(class_counts):
            counts = class_counts[breed]

            writer.writerow([
                breed,
                counts["train"],
                counts["validation"],
                counts["test"],
            ])

    print("\nClass distribution:")

    for breed in sorted(class_counts):
        c = class_counts[breed]

        print(
            f"{breed:28s} "
            f"train={c['train']:3d}, "
            f"val={c['validation']:3d}, "
            f"test={c['test']:3d}"
        )


def verify_splits(train, val, test):
    train_ids = {r["image_id"] for r in train}
    val_ids = {r["image_id"] for r in val}
    test_ids = {r["image_id"] for r in test}

    assert len(train_ids) == len(train)
    assert len(val_ids) == len(val)
    assert len(test_ids) == len(test)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    assert len(train) + len(val) == 3680
    assert len(test) == 3669

    train_labels = {r["label"] for r in train}
    val_labels = {r["label"] for r in val}
    test_labels = {r["label"] for r in test}

    assert train_labels == val_labels == test_labels == set(range(37))

    print("\nVerification passed!")
    print("No overlapping images.")
    print("All 37 classes exist in every split.")


def main():
    set_seed(SEED)

    print("=== Creating Dataset Splits ===\n")

    trainval = load_annotations("trainval")
    test = load_annotations("test")

    labels = [record["label"] for record in trainval]

    train, validation = train_test_split(
        trainval,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=labels,
    )

    save_split(train, "train.csv")
    save_split(validation, "validation.csv")
    save_split(test, "test.csv")

    verify_splits(train, validation, test)

    save_class_distribution(train, validation, test)

    config = {
        "seed": SEED,
        "validation_size": VALIDATION_SIZE,
        "method": "stratified train_test_split",
        "official_test_untouched": True,
        "train_images": len(train),
        "validation_images": len(validation),
        "test_images": len(test),
    }

    with open(SPLITS_DIR / "split_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print("\n=== Final Split Sizes ===")
    print("Train:", len(train))
    print("Validation:", len(validation))
    print("Test:", len(test))
    print("Total:", len(train) + len(validation) + len(test))

    print("\nDataset splitting completed successfully!")


if __name__ == "__main__":
    main()

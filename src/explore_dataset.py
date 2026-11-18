
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
from torchvision.datasets import OxfordIIITPet


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def load_dataset():
    print("Downloading/loading Oxford-IIIT Pet dataset...")

    trainval = OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="category",
        download=True,
    )

    test = OxfordIIITPet(
        root=str(DATA_DIR),
        split="test",
        target_types="category",
        download=True,
    )

    print("Dataset loaded successfully!")
    return trainval, test


def read_annotations(split_name):
    annotation_path = (
        DATA_DIR / "oxford-iiit-pet"
        / "annotations" / f"{split_name}.txt"
    )

    records = []

    with open(annotation_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            image_name = parts[0]
            breed_id = int(parts[1]) - 1

            breed_name = image_name.rsplit("_", 1)[0]

            records.append((breed_id, breed_name))

    return records


def save_class_counts(train_records, test_records):
    train_counts = {}
    test_counts = {}
    class_names = {}

    for label, breed in train_records:
        class_names[label] = breed
        train_counts[label] = train_counts.get(label, 0) + 1

    for label, breed in test_records:
        class_names[label] = breed
        test_counts[label] = test_counts.get(label, 0) + 1

    output_path = RESULTS_DIR / "dataset_class_counts.csv"

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["class_id", "breed", "trainval", "test"])

        for label in sorted(class_names):
            writer.writerow([
                label,
                class_names[label],
                train_counts.get(label, 0),
                test_counts.get(label, 0),
            ])

    print("\nTotal breeds:", len(class_names))
    print("\nBreed distribution:")

    for label in sorted(class_names):
        breed = class_names[label]
        print(
            f"{label:2d} | {breed:25s} | "
            f"trainval: {train_counts.get(label, 0):3d} | "
            f"test: {test_counts.get(label, 0):3d}"
        )

    with open(RESULTS_DIR / "class_names.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    return class_names


def visualize_samples(dataset, class_names):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    indices = [0, 100, 300, 500, 700, 900]

    for ax, index in zip(axes.flat, indices):
        image, label = dataset[index]

        ax.imshow(image)
        ax.set_title(class_names[label].replace("_", " "))
        ax.axis("off")

    plt.tight_layout()

    output_path = RESULTS_DIR / "dataset_samples.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("\nSample visualization saved:", output_path)


def main():
    trainval, test = load_dataset()

    print("\n=== Dataset Size ===")
    print("Official trainval:", len(trainval))
    print("Official test:", len(test))
    print("Total:", len(trainval) + len(test))

    train_records = read_annotations("trainval")
    test_records = read_annotations("test")

    assert len(train_records) == len(trainval)
    assert len(test_records) == len(test)

    class_names = save_class_counts(train_records, test_records)

    assert len(class_names) == 37
    assert set(class_names) == set(range(37))

    visualize_samples(trainval, class_names)

    print("\nDataset exploration completed successfully!")


if __name__ == "__main__":
    main()

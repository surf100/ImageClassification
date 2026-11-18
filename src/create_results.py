
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# =====================================
# 1. PATHS AND SETTINGS
# =====================================

BASE_DIR = Path(__file__).resolve().parent.parent

RESULTS_DIR = BASE_DIR / "results"
TEST_DIR = RESULTS_DIR / "test_evaluation"
OUTPUT_DIR = RESULTS_DIR / "report_figures"

SPLITS_DIR = BASE_DIR / "splits"
IMAGES_DIR = BASE_DIR / "data" / "oxford-iiit-pet" / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = [
    ("full_finetuning", "Full Fine-Tuning"),
    ("head_training", "Classifier-Head Training"),
    ("linear_svm", "Feature Extraction + SVM"),
]

NUM_CLASSES = 37


# =====================================
# 2. READ SAVED RESULTS
# =====================================

def read_csv(path):
    with open(path, "r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_comparison():
    test_rows = read_csv(TEST_DIR / "test_metrics.csv")

    test_metrics = {
        row["method"]: row
        for row in test_rows
    }

    full_config = read_json(
        RESULTS_DIR / "full_finetuning" / "training_config.json"
    )

    head_config = read_json(
        RESULTS_DIR / "head_training" / "training_config.json"
    )

    svm_config = read_json(
        RESULTS_DIR / "feature_extraction" / "training_config.json"
    )

    configs = {
        "full_finetuning": full_config,
        "head_training": head_config,
        "linear_svm": svm_config,
    }

    comparison = []

    for method_id, method_name in METHODS:
        metrics = test_metrics[method_id]
        config = configs[method_id]

        if method_id == "linear_svm":
            params = config["neural_network_trainable_parameters"]
            selected_time = config["total_selected_pipeline_seconds"]

            # Time spent fitting all candidate SVM classifiers
            search_time = sum(
                item["classifier_training_seconds"]
                for item in config["hyperparameter_search_results"]
            )

            total_search_time = (
                config["feature_extraction_seconds"] + search_time
            )
        else:
            params = config["trainable_parameters"]
            selected_time = config["training_time_seconds"]
            total_search_time = None

        comparison.append({
            "method": method_name,
            "test_accuracy": float(metrics["test_accuracy"]),
            "test_macro_f1": float(metrics["test_macro_f1"]),
            "trainable_nn_parameters": int(params),
            "selected_run_seconds": float(selected_time),
            "total_svm_search_seconds": total_search_time,
        })

    return comparison


# =====================================
# 3. SAVE COMPARISON TABLE
# =====================================

def save_comparison_table(comparison):
    output_path = OUTPUT_DIR / "comparison_table.csv"

    with open(
        output_path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(comparison[0].keys()),
        )

        writer.writeheader()
        writer.writerows(comparison)

    print("\n=== Final Results Table ===")

    for item in comparison:
        print(
            f"{item['method']:27s} | "
            f"Accuracy: {item['test_accuracy']:.4f} | "
            f"Macro F1: {item['test_macro_f1']:.4f} | "
            f"Parameters: {item['trainable_nn_parameters']:,} | "
            f"Time: {item['selected_run_seconds']:.2f}s"
        )


# =====================================
# 4. PERFORMANCE COMPARISON CHART
# =====================================

def plot_performance(comparison):
    names = [row["method"] for row in comparison]

    accuracies = [
        row["test_accuracy"] * 100
        for row in comparison
    ]

    f1_values = [
        row["test_macro_f1"] * 100
        for row in comparison
    ]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))

    bars_acc = ax.bar(
        x - width / 2,
        accuracies,
        width,
        label="Test Accuracy",
    )

    bars_f1 = ax.bar(
        x + width / 2,
        f1_values,
        width,
        label="Macro F1",
    )

    ax.bar_label(bars_acc, fmt="%.2f%%", padding=3)
    ax.bar_label(bars_f1, fmt="%.2f%%", padding=3)

    ax.set_ylabel("Score (%)")
    ax.set_title("Classification Performance on Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "performance_comparison.png",
        dpi=180,
    )
    plt.close()


# =====================================
# 5. COMPUTATIONAL COST CHART
# =====================================

def plot_computational_cost(comparison):
    names = [row["method"] for row in comparison]

    parameters = [
        row["trainable_nn_parameters"] / 1_000_000
        for row in comparison
    ]

    times = [
        row["selected_run_seconds"]
        for row in comparison
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = axes[0].bar(names, parameters)
    axes[0].set_title("Trainable Neural Network Parameters")
    axes[0].set_ylabel("Millions of parameters")
    axes[0].tick_params(axis="x", labelrotation=18)
    axes[0].bar_label(bars1, fmt="%.3f", padding=3)

    bars2 = axes[1].bar(names, times)
    axes[1].set_title("Recorded Training / Selected Pipeline Time")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", labelrotation=18)
    axes[1].bar_label(bars2, fmt="%.1f", padding=3)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "computational_cost.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


# =====================================
# 6. LOAD TEST IMAGE INFORMATION
# =====================================

def load_test_records():
    records = read_csv(SPLITS_DIR / "test.csv")

    assert len(records) == 3669
    return records


# =====================================
# 7. SELECT REAL PREDICTION EXAMPLES
# =====================================

def select_examples(predictions):
    correct = [
        row for row in predictions
        if int(row["correct"]) == 1
    ]

    incorrect = [
        row for row in predictions
        if int(row["correct"]) == 0
    ]

    # Choose varied true breeds when possible
    def choose_distinct(rows, amount=3):
        selected = []
        used_breeds = set()

        for row in rows:
            breed = row["true_breed"]

            if breed not in used_breeds:
                selected.append(row)
                used_breeds.add(breed)

            if len(selected) == amount:
                break

        return selected

    return choose_distinct(correct), choose_distinct(incorrect)


# =====================================
# 8. PLOT CORRECT / INCORRECT EXAMPLES
# =====================================

def plot_predictions(method_id, method_name, test_records):
    predictions = read_csv(
        TEST_DIR / f"{method_id}_predictions.csv"
    )

    assert len(predictions) == len(test_records)

    correct, incorrect = select_examples(predictions)

    assert len(correct) == 3
    assert len(incorrect) == 3

    selected = correct + incorrect

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for index, (ax, prediction) in enumerate(
        zip(axes.flat, selected)
    ):
        test_index = int(prediction["test_index"])
        record = test_records[test_index]

        assert int(record["label"]) == int(
            prediction["true_label"]
        )

        image_path = IMAGES_DIR / record["filename"]

        with Image.open(image_path) as image:
            ax.imshow(image.convert("RGB"))

        true_breed = prediction["true_breed"]
        predicted_breed = prediction["predicted_breed"]

        is_correct = index < 3

        status = "Correct" if is_correct else "Incorrect"

        ax.set_title(
            f"{status}\n"
            f"True: {true_breed}\n"
            f"Predicted: {predicted_breed}",
            fontsize=9,
        )

        ax.axis("off")

    fig.suptitle(
        f"{method_name}: Prediction Examples",
        fontsize=15,
    )

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"{method_id}_prediction_examples.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)

    return selected


# =====================================
# 9. SAVE SELECTED EXAMPLE INFORMATION
# =====================================

def save_example_records(all_examples):
    path = OUTPUT_DIR / "selected_examples.csv"

    with open(
        path, "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "method",
            "test_index",
            "true_breed",
            "predicted_breed",
            "correct",
        ])

        for method_id, examples in all_examples.items():
            for row in examples:
                writer.writerow([
                    method_id,
                    row["test_index"],
                    row["true_breed"],
                    row["predicted_breed"],
                    row["correct"],
                ])


# =====================================
# 10. MAIN
# =====================================

def main():
    print("=== Step 10: Results & Visualizations ===")

    comparison = load_comparison()

    save_comparison_table(comparison)
    plot_performance(comparison)
    plot_computational_cost(comparison)

    test_records = load_test_records()

    all_examples = {}

    for method_id, method_name in METHODS:
        print(f"\nCreating examples for {method_name}...")

        selected = plot_predictions(
            method_id,
            method_name,
            test_records,
        )

        all_examples[method_id] = selected

    save_example_records(all_examples)

    print("\n=== Files Created ===")

    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path.name)

    print("\nResults visualization completed successfully!")


if __name__ == "__main__":
    main()

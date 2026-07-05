from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision.models import efficientnet_b0

from prepare_data import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    build_split_dataset,
    get_preprocessing_config,
    get_transforms,
)
from train import ensure_output_dir, get_device

DEFAULT_DATA_DIR = Path("dataset")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "trained_model.pth"
DEFAULT_METRICS_PATH = DEFAULT_CHECKPOINT_DIR / "test_metrics.txt"
DEFAULT_CONFUSION_MATRIX_PATH = DEFAULT_CHECKPOINT_DIR / "test_confusion_matrices.png"
DEFAULT_PLOT_DPI = 150
EXPECTED_ARCHITECTURE = "efficientnet_b0"
EXPECTED_WEIGHTS = "EfficientNet_B0_Weights.DEFAULT"
CONFUSION_MATRIX_CELL_HEIGHT_RATIO = 2.0 / 3.0


# Parse runtime overrides while keeping evaluation defaults aligned with the repo workflow.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained EfficientNet-B0 rice leaf classifier.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--confusion-matrix-path", type=Path, default=DEFAULT_CONFUSION_MATRIX_PATH)
    parser.add_argument("--verify-images", action="store_true", default=False)
    return parser.parse_args()


# Load the shared dataset pipeline and return only the held-out test loader plus its class map.
def load_test_data(args: argparse.Namespace) -> tuple[torch.utils.data.DataLoader, Dict[str, int]]:
    test_dataset = build_split_dataset(
        data_dir=args.data_dir,
        split_name="test",
        transform=get_test_transform(),
        verify_images=args.verify_images,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    return test_dataloader, test_dataset.class_to_idx


# Expose the current test transform without building the train and validation datasets.
def get_test_transform():
    return get_transforms()["test"]


# Rebuild the saved EfficientNet-B0 classifier without downloading pretrained weights at evaluation time.
def build_evaluation_model(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# Load checkpoint metadata and fail clearly if the saved model does not match the current project contract.
def load_checkpoint(checkpoint_path: Path, expected_class_to_idx: Dict[str, int]) -> Dict[str, object]:
    if not checkpoint_path.exists():
        raise ValueError(f"Missing trained checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint payload is invalid: expected a metadata dictionary.")

    architecture = checkpoint.get("architecture")
    if architecture != EXPECTED_ARCHITECTURE:
        raise ValueError(
            f"Unsupported checkpoint architecture '{architecture}'. Expected '{EXPECTED_ARCHITECTURE}'."
        )

    saved_weights = checkpoint.get("weights")
    if saved_weights != EXPECTED_WEIGHTS:
        raise ValueError(
            f"Unsupported checkpoint weights '{saved_weights}'. Expected '{EXPECTED_WEIGHTS}'."
        )

    saved_class_to_idx = checkpoint.get("class_to_idx")
    if saved_class_to_idx != expected_class_to_idx:
        raise ValueError(
            "Checkpoint class mapping does not match the current evaluation class order. "
            f"Expected {expected_class_to_idx}, found {saved_class_to_idx}."
        )

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing 'model_state_dict'.")

    saved_preprocessing = checkpoint.get("preprocessing")
    current_preprocessing = get_preprocessing_config()
    if saved_preprocessing != current_preprocessing:
        raise ValueError(
            "Checkpoint preprocessing metadata does not match the current evaluation pipeline. "
            "Retrain the model or restore the preprocessing contract used to create this checkpoint."
        )

    return checkpoint


# Run the model across the full test loader and collect integer labels for metric calculation.
def collect_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[List[int], List[int]]:
    model.eval()
    all_targets: List[int] = []
    all_predictions: List[int] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1).cpu()

            all_targets.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())

    return all_targets, all_predictions


# Build a raw confusion matrix using the project’s fixed class order.
def build_confusion_matrix(
    targets: List[int],
    predictions: List[int],
    num_classes: int,
) -> torch.Tensor:
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    for target, prediction in zip(targets, predictions):
        confusion_matrix[target, prediction] += 1

    return confusion_matrix


# Derive overall and per-class classification metrics directly from the confusion matrix.
def compute_classification_metrics(
    confusion_matrix: torch.Tensor,
    class_names: List[str],
) -> Dict[str, object]:
    total_examples = int(confusion_matrix.sum().item())
    correct_predictions = int(confusion_matrix.diag().sum().item())
    overall_accuracy = correct_predictions / total_examples if total_examples > 0 else 0.0

    per_class_metrics: List[Dict[str, float | str]] = []
    for class_index, class_name in enumerate(class_names):
        true_positives = float(confusion_matrix[class_index, class_index].item())
        predicted_positives = float(confusion_matrix[:, class_index].sum().item())
        actual_positives = float(confusion_matrix[class_index, :].sum().item())

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        recall = true_positives / actual_positives if actual_positives > 0 else 0.0
        f1_score = 0.0
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)

        per_class_metrics.append(
            {
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "support": actual_positives,
            }
        )

    row_sums = confusion_matrix.sum(dim=1, keepdim=True).to(dtype=torch.float32)
    normalized_confusion_matrix = torch.where(
        row_sums > 0,
        confusion_matrix.to(dtype=torch.float32) / row_sums,
        torch.zeros_like(confusion_matrix, dtype=torch.float32),
    )

    return {
        "overall_accuracy": overall_accuracy,
        "per_class_metrics": per_class_metrics,
        "normalized_confusion_matrix": normalized_confusion_matrix,
    }


# Format a raw or normalized confusion matrix into a readable plain-text table.
def format_matrix(matrix: torch.Tensor, class_names: List[str], decimal_places: int) -> str:
    row_label = "true\\pred"
    row_label_width = max(len(row_label), *(len(class_name) for class_name in class_names))
    formatted_rows = []

    if matrix.dtype.is_floating_point:
        value_strings = [
            [f"{value:.{decimal_places}f}" for value in row]
            for row in matrix.tolist()
        ]
    else:
        value_strings = [
            [str(int(value)) for value in row]
            for row in matrix.tolist()
        ]

    value_width = max(len(class_name) for class_name in class_names)
    for row in value_strings:
        value_width = max(value_width, *(len(value) for value in row))

    header = f"{row_label:<{row_label_width}} " + " ".join(
        f"{class_name:>{value_width}}" for class_name in class_names
    )
    formatted_rows.append(header)

    for class_name, row in zip(class_names, value_strings):
        formatted_rows.append(
            f"{class_name:<{row_label_width}} " + " ".join(f"{value:>{value_width}}" for value in row)
        )

    return "\n".join(formatted_rows)


# Convert the computed metrics and matrices into a report suitable for the terminal and text file output.
def format_metrics_report(
    metrics: Dict[str, object],
    raw_confusion_matrix: torch.Tensor,
    class_names: List[str],
) -> str:
    lines = [f"Overall accuracy: {metrics['overall_accuracy']:.4f}", "", "Per-class metrics:"]
    lines.append(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1 Score':>10} {'Support':>10}")

    per_class_metrics = metrics["per_class_metrics"]
    if not isinstance(per_class_metrics, list):
        raise ValueError("Expected per-class metrics to be a list.")

    for class_metrics in per_class_metrics:
        if not isinstance(class_metrics, dict):
            raise ValueError("Expected each class metric entry to be a dictionary.")
        lines.append(
            f"{str(class_metrics['class_name']):<12} "
            f"{float(class_metrics['precision']):>10.4f} "
            f"{float(class_metrics['recall']):>10.4f} "
            f"{float(class_metrics['f1_score']):>10.4f} "
            f"{int(float(class_metrics['support'])):>10}"
        )

    normalized_confusion_matrix = metrics["normalized_confusion_matrix"]
    if not isinstance(normalized_confusion_matrix, torch.Tensor):
        raise ValueError("Expected normalized confusion matrix to be a tensor.")

    lines.extend(
        [
            "",
            "Raw confusion matrix:",
            format_matrix(raw_confusion_matrix, class_names, decimal_places=0),
            "",
            "Normalized confusion matrix:",
            format_matrix(normalized_confusion_matrix, class_names, decimal_places=4),
        ]
    )
    return "\n".join(lines)


# Save the metrics report as a plain-text artifact in the checkpoint directory.
def save_metrics_report(report: str, metrics_path: Path) -> None:
    ensure_output_dir(metrics_path.parent)
    metrics_path.write_text(f"{report}\n", encoding="utf-8")


# Scale figure and label sizes so the saved matrix plot remains readable as class counts grow.
def get_confusion_matrix_layout(class_count: int) -> Dict[str, float]:
    plot_width = max(10.0, class_count * 1.2)
    per_matrix_height = max(4.5, class_count * 0.8)
    tick_font_size = max(8.0, 12.0 - max(0, class_count - 5) * 0.4)
    annotation_font_size = max(6.0, 11.0 - max(0, class_count - 5) * 0.5)

    return {
        "plot_width": plot_width,
        "per_matrix_height": per_matrix_height,
        "tick_font_size": tick_font_size,
        "annotation_font_size": annotation_font_size,
    }


# Render one figure with raw and normalized confusion matrices and save it to disk.
def save_confusion_matrix_plot(
    raw_confusion_matrix: torch.Tensor,
    normalized_confusion_matrix: torch.Tensor,
    class_names: List[str],
    output_path: Path,
) -> None:
    ensure_output_dir(output_path.parent)
    layout = get_confusion_matrix_layout(len(class_names))
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(layout["plot_width"], layout["per_matrix_height"] * 2),
        constrained_layout=True,
    )
    figure.set_constrained_layout_pads(hspace=0.08, h_pad=0.12)
    matrices = [
        (raw_confusion_matrix.to(dtype=torch.float32), "Raw Confusion Matrix", "Count", ".0f"),
        (normalized_confusion_matrix, "Normalized Confusion Matrix", "Proportion", ".2f"),
    ]

    for axis, (matrix, title, colorbar_label, value_format) in zip(axes, matrices):
        image = axis.imshow(matrix.tolist(), cmap="Blues")
        axis.set_aspect(CONFUSION_MATRIX_CELL_HEIGHT_RATIO)
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        axis.set_yticks(range(len(class_names)), class_names)
        axis.tick_params(axis="x", labelsize=layout["tick_font_size"])
        axis.tick_params(axis="y", labelsize=layout["tick_font_size"])

        colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        colorbar.ax.set_ylabel(colorbar_label, rotation=270, labelpad=15)
        colorbar.ax.tick_params(labelsize=layout["tick_font_size"])

        matrix_max = float(matrix.max().item()) if matrix.numel() > 0 else 0.0
        text_threshold = matrix_max / 2 if matrix_max > 0 else 0.0

        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = float(matrix[row_index, column_index].item())
                axis.text(
                    column_index,
                    row_index,
                    format(value, value_format),
                    ha="center",
                    va="center",
                    color="white" if value > text_threshold else "black",
                    fontsize=layout["annotation_font_size"],
                )

    figure.savefig(output_path, dpi=DEFAULT_PLOT_DPI, bbox_inches="tight")
    plt.close(figure)


# Execute the full evaluation workflow and persist both the text and image outputs.
def evaluate_model(args: argparse.Namespace) -> Dict[str, Path | str]:
    device = get_device()
    test_dataloader, class_to_idx = load_test_data(args)
    checkpoint = load_checkpoint(args.checkpoint_path, class_to_idx)
    class_names = list(class_to_idx.keys())

    model = build_evaluation_model(num_classes=len(class_names)).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as error:
        raise ValueError(f"Checkpoint weights are incompatible with the evaluation model: {error}") from error

    targets, predictions = collect_predictions(model, test_dataloader, device)
    raw_confusion_matrix = build_confusion_matrix(targets, predictions, num_classes=len(class_names))
    metrics = compute_classification_metrics(raw_confusion_matrix, class_names)
    report = format_metrics_report(metrics, raw_confusion_matrix, class_names)

    save_metrics_report(report, args.metrics_path)
    normalized_confusion_matrix = metrics["normalized_confusion_matrix"]
    if not isinstance(normalized_confusion_matrix, torch.Tensor):
        raise ValueError("Expected normalized confusion matrix to be a tensor.")
    save_confusion_matrix_plot(
        raw_confusion_matrix=raw_confusion_matrix,
        normalized_confusion_matrix=normalized_confusion_matrix,
        class_names=class_names,
        output_path=args.confusion_matrix_path,
    )

    print(f"Using device: {device}")
    print(report)
    print()
    print(f"Saved metrics report to {args.metrics_path}")
    print(f"Saved confusion matrix plot to {args.confusion_matrix_path}")

    return {
        "metrics_path": args.metrics_path,
        "confusion_matrix_path": args.confusion_matrix_path,
        "report": report,
    }


# Provide the script entry point for local evaluation runs.
def main() -> None:
    args = parse_args()
    evaluate_model(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from prepare_data import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_SEED,
    create_dataloaders,
    get_preprocessing_config,
)

DEFAULT_DATA_DIR = Path("dataset")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "trained_model.pth"
DEFAULT_HYPERPARAMS_REPORT_PATH = DEFAULT_CHECKPOINT_DIR / "hyperparams.txt"
DEFAULT_EPOCHS = 30
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_UNFREEZE_BLOCKS = 4
DEFAULT_PLOT_DPI = 150


# Reject non-positive epoch counts before training can produce misleading artifacts.
def parse_positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value < 1:
        raise argparse.ArgumentTypeError("epochs must be at least 1")
    return parsed_value


# Parse optional runtime overrides while keeping sensible training defaults in code.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an EfficientNet-B0 rice leaf classifier.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=parse_positive_int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--verify-images", action="store_true", default=False)
    return parser.parse_args()


# Set reproducible seeds so the train and validation splits remain stable across runs.
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Pick the best available device so the same script works on both GPU and CPU machines.
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Create the checkpoint directory up front so model files and plots can always be written.
def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


# Load the prepared train and validation loaders from the existing data module.
def load_training_data(args: argparse.Namespace) -> tuple[Dict[str, torch.utils.data.DataLoader], Dict[str, int]]:
    dataloaders, class_to_idx = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        verify_images=args.verify_images,
    )
    return dataloaders, class_to_idx


# Build a pretrained EfficientNet-B0 and retarget its classifier head to the project classes.
def build_model(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for block in model.features[-DEFAULT_UNFREEZE_BLOCKS:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return model


# Construct the optimizer and scheduler for the subset of parameters that are trainable.
def build_optimization_components(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, ReduceLROnPlateau]:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    return optimizer, scheduler


# Run one phase of the loop and return the average loss and accuracy for that pass.
def run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    running_loss = 0.0
    running_correct = 0
    running_examples = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (predictions == labels).sum().item()
        running_examples += batch_size

    epoch_loss = running_loss / running_examples
    epoch_accuracy = running_correct / running_examples
    return epoch_loss, epoch_accuracy


# Save the strongest validation checkpoint together with enough metadata to reload it later.
def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    class_to_idx: Dict[str, int],
    best_val_accuracy: float,
    args: argparse.Namespace,
    preprocessing_config: Dict[str, object],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_to_idx": class_to_idx,
            "best_val_accuracy": best_val_accuracy,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "architecture": "efficientnet_b0",
            "weights": "EfficientNet_B0_Weights.DEFAULT",
            "preprocessing": preprocessing_config,
        },
        checkpoint_path,
    )


# Draw and save the final loss and accuracy graphs from the collected history.
def save_training_plots(history: Dict[str, List[float]], output_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_graph.png", dpi=DEFAULT_PLOT_DPI)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_graph.png", dpi=DEFAULT_PLOT_DPI)
    plt.close()


# Print a compact per-epoch summary so progress is visible while the model trains.
def log_epoch_metrics(
    epoch_index: int,
    total_epochs: int,
    train_loss: float,
    train_accuracy: float,
    val_loss: float,
    val_accuracy: float,
    best: bool,
) -> None:
    print(
        f"Epoch {epoch_index}/{total_epochs} | "
        f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
        f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} | "
        f"best={'true' if best else 'false'}"
    )


# Build the active training setup report before the first epoch so runs are easy to audit later.
def build_training_configuration_report(
    args: argparse.Namespace,
    class_to_idx: Dict[str, int],
    preprocessing_config: Dict[str, object],
) -> str:
    train_preset = preprocessing_config["train"]
    lines = ["Current hyperparameters:"]

    for name, value in vars(args).items():
        lines.append(f"  {name}: {value}")
    lines.append(f"  unfreeze_blocks: {DEFAULT_UNFREEZE_BLOCKS}")
    lines.append(f"  class_labels: {list(class_to_idx.keys())}")
    lines.append(
        "  train_transform_preset: "
        f"{train_preset['active_preset_id']} ({train_preset['active_preset_name']})"
    )
    return "\n".join(lines)


# Persist the training setup report so each run keeps a readable record next to its checkpoint.
def save_training_configuration_report(
    report_path: Path,
    args: argparse.Namespace,
    class_to_idx: Dict[str, int],
    preprocessing_config: Dict[str, object],
) -> None:
    report = build_training_configuration_report(args, class_to_idx, preprocessing_config)
    report_path.write_text(f"{report}\n", encoding="utf-8")


# Print the saved training setup report so the run still announces its configuration in the terminal.
def log_training_configuration(
    args: argparse.Namespace,
    class_to_idx: Dict[str, int],
    preprocessing_config: Dict[str, object],
) -> None:
    print(build_training_configuration_report(args, class_to_idx, preprocessing_config))


# Execute the full transfer-learning workflow and persist the best validation result.
def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    checkpoint_dir = args.checkpoint_dir
    checkpoint_path = checkpoint_dir / DEFAULT_CHECKPOINT_PATH.name
    hyperparams_report_path = checkpoint_dir / DEFAULT_HYPERPARAMS_REPORT_PATH.name
    ensure_output_dir(checkpoint_dir)

    print(f"Using device: {device}")
    dataloaders, class_to_idx = load_training_data(args)
    preprocessing_config = get_preprocessing_config()
    save_training_configuration_report(
        hyperparams_report_path,
        args,
        class_to_idx,
        preprocessing_config,
    )
    log_training_configuration(args, class_to_idx, preprocessing_config)
    model = build_model(num_classes=len(class_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = build_optimization_components(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }
    best_val_accuracy = float("-inf")
    best_epoch: int | None = None
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch_index in range(1, args.epochs + 1):
        best = False   
        train_loss, train_accuracy = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )
        val_loss, val_accuracy = run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            optimizer=None,
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best = True
            best_val_accuracy = val_accuracy
            best_epoch = epoch_index
            best_state_dict = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state_dict)
            save_checkpoint(
                checkpoint_path,
                model,
                class_to_idx,
                best_val_accuracy,
                args,
                preprocessing_config,
            )
        
        log_epoch_metrics(epoch_index, args.epochs, train_loss, train_accuracy, val_loss, val_accuracy, best)

    model.load_state_dict(best_state_dict)
    save_checkpoint(
        checkpoint_path,
        model,
        class_to_idx,
        best_val_accuracy,
        args,
        preprocessing_config,
    )
    save_training_plots(history, checkpoint_dir)

    saved_epoch = best_epoch if best_epoch is not None else args.epochs
    print(f"Saved best checkpoint from epoch {saved_epoch} to {checkpoint_path}")
    print(f"Saved loss graph to {checkpoint_dir / 'loss_graph.png'}")
    print(f"Saved accuracy graph to {checkpoint_dir / 'accuracy_graph.png'}")


# Provide the script entry point for local training runs.
def main() -> None:
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()

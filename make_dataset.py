from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

RAW_DATA_DIR = Path("data")
DATASET_DIR = Path("dataset")
DEFAULT_SEED = 42
SPLIT_RATIOS: Dict[str, float] = {"train": 0.6, "val": 0.2, "test": 0.2}
SPLIT_ORDER: Tuple[str, ...] = ("train", "val", "test")
SPLIT_FOLDERS: Dict[str, str] = {
    "train": "train_images",
    "val": "val_images",
    "test": "test_images",
}
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")


# Return only class directories from the raw data root while ignoring files like train.csv.
def list_class_directories(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise ValueError(f"Raw data directory does not exist: {data_dir}")

    class_directories = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not class_directories:
        raise ValueError(f"No class directories found in {data_dir}")

    return class_directories


# Collect image files from one class directory using a stable ordering before shuffling.
def list_image_files(class_dir: Path) -> List[Path]:
    image_files = sorted(
        path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(image_files) < len(SPLIT_ORDER):
        raise ValueError(
            f"Class '{class_dir.name}' has only {len(image_files)} image(s). "
            "At least 3 are required to keep train, val, and test non-empty."
        )

    return image_files


# Turn a class image count into non-empty train, validation, and test split sizes.
def split_counts(total_items: int) -> Tuple[int, int, int]:
    counts = {split_name: 1 for split_name in SPLIT_ORDER}
    remaining_items = total_items - len(SPLIT_ORDER)

    if remaining_items == 0:
        return counts["train"], counts["val"], counts["test"]

    raw_extras = {
        split_name: remaining_items * SPLIT_RATIOS[split_name] for split_name in SPLIT_ORDER
    }
    extra_counts = {split_name: int(raw_extras[split_name]) for split_name in SPLIT_ORDER}

    for split_name in SPLIT_ORDER:
        counts[split_name] += extra_counts[split_name]

    leftover = remaining_items - sum(extra_counts.values())
    ranked_splits = sorted(
        SPLIT_ORDER,
        key=lambda split_name: (raw_extras[split_name] - extra_counts[split_name], SPLIT_RATIOS[split_name]),
        reverse=True,
    )

    for split_name in ranked_splits[:leftover]:
        counts[split_name] += 1

    return counts["train"], counts["val"], counts["test"]


# Deterministically assign a class's images into train, validation, and test lists.
def split_class_images(image_files: Iterable[Path], seed: int) -> Dict[str, List[Path]]:
    shuffled_files = list(image_files)
    rng = random.Random(seed)
    rng.shuffle(shuffled_files)

    train_count, val_count, _ = split_counts(len(shuffled_files))
    return {
        "train": shuffled_files[:train_count],
        "val": shuffled_files[train_count : train_count + val_count],
        "test": shuffled_files[train_count + val_count :],
    }


# Recreate a target dataset root so each build writes into a clean directory tree.
def reset_dataset_root(dataset_dir: Path) -> None:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    for split_folder in SPLIT_FOLDERS.values():
        (dataset_dir / split_folder).mkdir(parents=True, exist_ok=True)


# Copy one class's assigned files into its generated split directories.
def copy_split_files(dataset_dir: Path, class_name: str, split_map: Dict[str, List[Path]]) -> Dict[str, int]:
    split_counts_map: Dict[str, int] = {}

    for split_name, image_files in split_map.items():
        target_dir = dataset_dir / SPLIT_FOLDERS[split_name] / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for image_path in image_files:
            shutil.copy2(image_path, target_dir / image_path.name)

        split_counts_map[split_name] = len(image_files)

    return split_counts_map


# Validate every class and compute its split plan before touching the live dataset directory.
def plan_dataset_splits(class_directories: List[Path], seed: int) -> Dict[str, Dict[str, List[Path]]]:
    dataset_plan: Dict[str, Dict[str, List[Path]]] = {}

    for class_index, class_dir in enumerate(class_directories):
        image_files = list_image_files(class_dir)
        dataset_plan[class_dir.name] = split_class_images(image_files, seed=seed + class_index)

    return dataset_plan


# Materialize a validated dataset plan into a fully built temporary directory.
def write_dataset_plan(dataset_dir: Path, dataset_plan: Dict[str, Dict[str, List[Path]]]) -> Dict[str, Dict[str, int]]:
    reset_dataset_root(dataset_dir)

    dataset_summary: Dict[str, Dict[str, int]] = {}
    for class_name, split_map in dataset_plan.items():
        dataset_summary[class_name] = copy_split_files(dataset_dir, class_name, split_map)

    return dataset_summary


# Swap the completed temporary dataset into place without exposing a partial build on failure.
def finalize_dataset_swap(temp_dataset_dir: Path, dataset_dir: Path) -> None:
    backup_dataset_dir = dataset_dir.with_name(f"{dataset_dir.name}_backup")
    if backup_dataset_dir.exists():
        shutil.rmtree(backup_dataset_dir)

    if dataset_dir.exists():
        dataset_dir.rename(backup_dataset_dir)

    try:
        temp_dataset_dir.rename(dataset_dir)
    except Exception:
        if backup_dataset_dir.exists() and not dataset_dir.exists():
            backup_dataset_dir.rename(dataset_dir)
        raise
    else:
        if backup_dataset_dir.exists():
            shutil.rmtree(backup_dataset_dir)


# Generate deterministic dataset folders for every class found under the raw data directory.
def build_dataset(
    data_dir: Path = RAW_DATA_DIR,
    dataset_dir: Path = DATASET_DIR,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Dict[str, int]]:
    class_directories = list_class_directories(data_dir)
    dataset_plan = plan_dataset_splits(class_directories, seed=seed)
    temp_dataset_dir = dataset_dir.with_name(f"{dataset_dir.name}_tmp")

    try:
        dataset_summary = write_dataset_plan(temp_dataset_dir, dataset_plan)
        finalize_dataset_swap(temp_dataset_dir, dataset_dir)
        return dataset_summary
    finally:
        if temp_dataset_dir.exists():
            shutil.rmtree(temp_dataset_dir)


# Format the generated split counts so a human can quickly verify the output.
def summarize_dataset(dataset_summary: Dict[str, Dict[str, int]]) -> str:
    lines: List[str] = []
    for class_name, split_counts_map in dataset_summary.items():
        counts_text = ", ".join(
            f"{split_name}={split_counts_map[split_name]}" for split_name in SPLIT_ORDER
        )
        lines.append(f"{class_name}: {counts_text}")
    return "\n".join(lines)


# Provide the command-line entry point for generating the dataset split folders.
def main() -> None:
    dataset_summary = build_dataset()
    print("Generated dataset splits:")
    print(summarize_dataset(dataset_summary))


if __name__ == "__main__":
    main()

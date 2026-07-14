from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import imagehash
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import GroupShuffleSplit

RAW_DATA_DIR = Path("data")
DATASET_DIR = Path("dataset")
DEFAULT_SEED = 42
DEFAULT_HASH_THRESHOLD = 5
SPLIT_RATIOS: Dict[str, float] = {"train": 0.6, "val": 0.2, "test": 0.2}
SPLIT_ORDER: Tuple[str, ...] = ("train", "val", "test")
SPLIT_FOLDERS: Dict[str, str] = {
    "train": "train_images",
    "val": "val_images",
    "test": "test_images",
}
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")
GROUP_SPLIT_ATTEMPTS = 32


@dataclass(frozen=True)
class ImageFingerprint:
    path: Path
    content_hash: str
    perceptual_hash: imagehash.ImageHash
    difference_hash: imagehash.ImageHash


# Return only class directories from the raw data root while ignoring files like train.csv.
def list_class_directories(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        raise ValueError(f"Raw data directory does not exist: {data_dir}")

    class_directories = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not class_directories:
        raise ValueError(f"No class directories found in {data_dir}")

    return class_directories


# Validate the inclusive Hamming-distance threshold before any grouping work begins.
def validate_similarity_threshold(similarity_threshold: int) -> None:
    if not 0 <= similarity_threshold <= 64:
        raise ValueError(
            f"Similarity threshold must be between 0 and 64 inclusive; received {similarity_threshold}."
        )


# Compute a stable content hash so exact duplicates join the same similarity group.
def hash_file_contents(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Load deterministic image fingerprints for one class and fail early on unreadable files.
def load_image_fingerprints(class_dir: Path) -> List[ImageFingerprint]:
    image_files = sorted(
        path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        raise ValueError(f"Class '{class_dir.name}' has no supported image files to split.")

    fingerprints: List[ImageFingerprint] = []
    for image_path in image_files:
        try:
            with Image.open(image_path) as image:
                normalized_image = image.convert("RGB")
                perceptual_hash = imagehash.phash(normalized_image)
                difference_hash = imagehash.dhash(normalized_image)
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise ValueError(f"Unable to decode image '{image_path}': {error}") from error

        fingerprints.append(
            ImageFingerprint(
                path=image_path,
                content_hash=hash_file_contents(image_path),
                perceptual_hash=perceptual_hash,
                difference_hash=difference_hash,
            )
        )

    return fingerprints


# Decide whether two same-class images belong to the same near-duplicate cluster.
def fingerprints_are_similar(
    first_fingerprint: ImageFingerprint,
    second_fingerprint: ImageFingerprint,
    similarity_threshold: int,
) -> bool:
    if first_fingerprint.content_hash == second_fingerprint.content_hash:
        return True

    perceptual_distance = first_fingerprint.perceptual_hash - second_fingerprint.perceptual_hash
    difference_distance = first_fingerprint.difference_hash - second_fingerprint.difference_hash
    return max(perceptual_distance, difference_distance) <= similarity_threshold


# Build stable connected components so transitive matches share one group identifier.
def build_group_map(
    class_dir: Path,
    fingerprints: Sequence[ImageFingerprint],
    similarity_threshold: int,
) -> Dict[str, List[Path]]:
    parent = list(range(len(fingerprints)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first_index: int, second_index: int) -> None:
        first_root = find(first_index)
        second_root = find(second_index)
        if first_root != second_root:
            parent[second_root] = first_root

    for first_index, first_fingerprint in enumerate(fingerprints):
        for second_index in range(first_index + 1, len(fingerprints)):
            second_fingerprint = fingerprints[second_index]
            if fingerprints_are_similar(first_fingerprint, second_fingerprint, similarity_threshold):
                union(first_index, second_index)

    grouped_paths: Dict[int, List[Path]] = {}
    for index, fingerprint in enumerate(fingerprints):
        grouped_paths.setdefault(find(index), []).append(fingerprint.path)

    ordered_groups = sorted(
        (sorted(group_paths) for group_paths in grouped_paths.values()),
        key=lambda group_paths: tuple(path.name for path in group_paths),
    )

    return {
        f"{class_dir.name}_group_{group_index:04d}": group_paths
        for group_index, group_paths in enumerate(ordered_groups, start=1)
    }


# Score a split candidate by how closely its file counts match the configured ratios.
def split_score(split_map: Dict[str, List[str]], group_map: Dict[str, List[Path]]) -> float:
    total_images = sum(len(group_paths) for group_paths in group_map.values())
    actual_counts = {
        split_name: sum(len(group_map[group_id]) for group_id in split_map[split_name])
        for split_name in SPLIT_ORDER
    }
    return sum(
        abs(actual_counts[split_name] - (total_images * SPLIT_RATIOS[split_name])) for split_name in SPLIT_ORDER
    )


# Split group identifiers with a deterministic search over group-aware shuffle candidates.
def split_group_ids(class_name: str, group_map: Dict[str, List[Path]], seed: int) -> Dict[str, List[str]]:
    ordered_group_ids = sorted(group_map)
    if len(ordered_group_ids) < len(SPLIT_ORDER):
        raise ValueError(
            f"Class '{class_name}' has only {len(ordered_group_ids)} near-duplicate group(s). "
            "At least 3 groups are required to keep train, val, and test non-empty."
        )

    samples = [[group_id] for group_id in ordered_group_ids]
    best_split: Dict[str, List[str]] | None = None
    best_score: float | None = None

    for attempt in range(GROUP_SPLIT_ATTEMPTS):
        try:
            outer_splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=SPLIT_RATIOS["test"],
                random_state=seed + attempt,
            )
            train_val_indices, test_indices = next(
                outer_splitter.split(samples, groups=ordered_group_ids)
            )
        except ValueError:
            continue

        train_val_group_ids = [ordered_group_ids[index] for index in train_val_indices]
        test_group_ids = [ordered_group_ids[index] for index in test_indices]
        if not train_val_group_ids or not test_group_ids:
            continue

        inner_samples = [[group_id] for group_id in train_val_group_ids]
        try:
            inner_splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=SPLIT_RATIOS["val"] / (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]),
                random_state=seed + GROUP_SPLIT_ATTEMPTS + attempt,
            )
            train_indices, val_indices = next(
                inner_splitter.split(inner_samples, groups=train_val_group_ids)
            )
        except ValueError:
            continue

        split_map = {
            "train": [train_val_group_ids[index] for index in train_indices],
            "val": [train_val_group_ids[index] for index in val_indices],
            "test": test_group_ids,
        }
        if any(not split_map[split_name] for split_name in SPLIT_ORDER):
            continue

        candidate_score = split_score(split_map, group_map)
        if best_split is None or candidate_score < best_score:
            best_split = split_map
            best_score = candidate_score

    if best_split is None:
        raise ValueError(
            f"Unable to create a non-empty group-aware split for class '{class_name}' "
            "while preserving group boundaries."
        )

    return {split_name: sorted(best_split[split_name]) for split_name in SPLIT_ORDER}


# Expand a group-level split plan back into deterministic image-path assignments.
def split_class_images(class_dir: Path, seed: int, similarity_threshold: int) -> Dict[str, List[Path]]:
    fingerprints = load_image_fingerprints(class_dir)
    group_map = build_group_map(class_dir, fingerprints, similarity_threshold)
    split_groups = split_group_ids(class_dir.name, group_map, seed)

    split_map: Dict[str, List[Path]] = {}
    for split_name in SPLIT_ORDER:
        split_paths = [
            image_path
            for group_id in split_groups[split_name]
            for image_path in group_map[group_id]
        ]
        split_map[split_name] = sorted(split_paths)

    return split_map


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
def plan_dataset_splits(
    class_directories: List[Path],
    seed: int,
    similarity_threshold: int,
) -> Dict[str, Dict[str, List[Path]]]:
    validate_similarity_threshold(similarity_threshold)
    dataset_plan: Dict[str, Dict[str, List[Path]]] = {}

    for class_index, class_dir in enumerate(class_directories):
        dataset_plan[class_dir.name] = split_class_images(
            class_dir,
            seed=seed + class_index,
            similarity_threshold=similarity_threshold,
        )

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
    similarity_threshold: int = DEFAULT_HASH_THRESHOLD,
) -> Dict[str, Dict[str, int]]:
    class_directories = list_class_directories(data_dir)
    dataset_plan = plan_dataset_splits(
        class_directories,
        seed=seed,
        similarity_threshold=similarity_threshold,
    )
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

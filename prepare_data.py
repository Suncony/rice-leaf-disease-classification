from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights
from torchvision.transforms import InterpolationMode, v2

ALLOWED_CLASSES: Tuple[str, ...] = ("normal", "blast", "brown_spot")
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0
DEFAULT_SEED = 42
SPLIT_ORDER: Tuple[str, ...] = ("train", "val", "test")
SPLIT_ROOTS: Dict[str, str] = {
    "train": "train_images",
    "val": "val_images",
    "test": "test_images",
}
TRAIN_CROP_SCALE: Tuple[float, float] = (0.9, 1.0)
TRAIN_HORIZONTAL_FLIP_PROBABILITY = 0.5
INPUT_IMAGE_SIZE = 224
EVAL_RESIZE_SIZE = 256


class FilteredImageFolderSubset(Dataset):
    def __init__(
        self,
        base_dataset: ImageFolder,
        indices: Sequence[int],
        target_map: Dict[int, int],
        transform: v2.Transform | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.target_map = target_map
        self.transform = transform
        self.samples = [base_dataset.samples[index] for index in self.indices]
        self.targets = [self.target_map[base_dataset.targets[index]] for index in self.indices]
        self.class_to_idx = {class_name: index for index, class_name in enumerate(ALLOWED_CLASSES)}
        self.classes = list(ALLOWED_CLASSES)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        sample_index = self.indices[item]
        image_path, target = self.base_dataset.samples[sample_index]
        image = self.base_dataset.loader(image_path)
        target = self.target_map[target]

        if self.transform is not None:
            image = self.transform(image)

        return image, target


# Build the training and evaluation transforms from the pretrained EfficientNet contract.
def get_transforms() -> Dict[str, v2.Compose]:
    weights = EfficientNet_B0_Weights.DEFAULT
    pretrained_transforms = weights.transforms()
    normalize = v2.Normalize(mean=pretrained_transforms.mean, std=pretrained_transforms.std)

    train_transform = v2.Compose(
        [
            v2.RandomResizedCrop(
                size=INPUT_IMAGE_SIZE,
                scale=TRAIN_CROP_SCALE,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=TRAIN_HORIZONTAL_FLIP_PROBABILITY),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            normalize,
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.Resize(size=EVAL_RESIZE_SIZE, interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(size=INPUT_IMAGE_SIZE),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            normalize,
        ]
    )

    return {"train": train_transform, "val": eval_transform, "test": eval_transform}


# Expose the preprocessing metadata that training checkpoints need to stay self-describing.
def get_preprocessing_config() -> Dict[str, object]:
    weights = EfficientNet_B0_Weights.DEFAULT
    pretrained_transforms = weights.transforms()

    return {
        "weights_enum": "EfficientNet_B0_Weights.DEFAULT",
        "input_dtype": "torch.float32",
        "input_scale_range": [0.0, 1.0],
        "normalization_mean": list(pretrained_transforms.mean),
        "normalization_std": list(pretrained_transforms.std),
        "train": {
            "random_resized_crop_size": INPUT_IMAGE_SIZE,
            "random_resized_crop_scale": list(TRAIN_CROP_SCALE),
            "horizontal_flip_probability": TRAIN_HORIZONTAL_FLIP_PROBABILITY,
            "interpolation": InterpolationMode.BICUBIC.name,
            "antialias": True,
        },
        "eval": {
            "resize_size": EVAL_RESIZE_SIZE,
            "center_crop_size": INPUT_IMAGE_SIZE,
            "interpolation": InterpolationMode.BICUBIC.name,
            "antialias": True,
        },
    }


# Resolve the generated split folder for one dataset partition.
def get_split_root(data_dir: str | Path, split_name: str) -> Path:
    if split_name not in SPLIT_ROOTS:
        supported = ", ".join(SPLIT_ORDER)
        raise ValueError(f"Unsupported split '{split_name}'. Expected one of: {supported}")

    return Path(data_dir) / SPLIT_ROOTS[split_name]


# Validate that a generated split folder contains at least one class directory before loading it.
def validate_split_root_contents(split_root: Path) -> None:
    class_directories = [path for path in split_root.iterdir() if path.is_dir()]
    if not class_directories:
        raise ValueError(
            f"Dataset split folder is empty or incomplete: {split_root}. "
            "Run make_dataset.py again to generate dataset/train_images, dataset/val_images, and dataset/test_images."
        )


# Load one generated split folder and verify that every required training class is present.
def load_split_dataset(data_dir: str | Path, split_name: str) -> ImageFolder:
    split_root = get_split_root(data_dir, split_name)
    if not split_root.exists():
        raise ValueError(
            f"Missing dataset split folder: {split_root}. "
            "Run make_dataset.py to generate dataset/train_images, dataset/val_images, and dataset/test_images."
        )

    validate_split_root_contents(split_root)
    dataset = ImageFolder(root=str(split_root))
    missing_classes = [class_name for class_name in ALLOWED_CLASSES if class_name not in dataset.class_to_idx]
    if missing_classes:
        missing = ", ".join(sorted(missing_classes))
        raise ValueError(f"Missing required class folders in {split_root}: {missing}")

    return dataset


# Select only the currently supported training classes from a split that may contain additional labels.
def filter_indices(dataset: ImageFolder, class_names: Iterable[str] = ALLOWED_CLASSES) -> List[int]:
    allowed_target_ids = {dataset.class_to_idx[class_name] for class_name in class_names}
    filtered_indices = [
        index for index, target in enumerate(dataset.targets) if target in allowed_target_ids
    ]

    if not filtered_indices:
        allowed = ", ".join(class_names)
        raise ValueError(f"No images found for the requested classes: {allowed}")

    return filtered_indices


# Seed worker processes deterministically so shuffled training batches stay reproducible.
def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


# Verify that each selected image file can actually be opened before model training starts.
def validate_class_images(dataset: ImageFolder, filtered_indices: Sequence[int]) -> None:
    invalid_files: List[str] = []

    for index in filtered_indices:
        image_path, _ = dataset.samples[index]
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as error:
            invalid_files.append(f"{image_path} ({error})")

    if invalid_files:
        formatted_files = "\n".join(invalid_files[:5])
        remainder = len(invalid_files) - 5
        if remainder > 0:
            formatted_files = f"{formatted_files}\n... and {remainder} more"
        raise ValueError(f"Found unreadable image files during dataset preparation:\n{formatted_files}")


# Convert one split folder into a filtered dataset that preserves the project's fixed class order.
def build_split_dataset(
    data_dir: str | Path,
    split_name: str,
    transform: v2.Transform | None,
    verify_images: bool,
) -> FilteredImageFolderSubset:
    base_dataset = load_split_dataset(data_dir, split_name)
    filtered_indices = filter_indices(base_dataset)
    if verify_images:
        validate_class_images(base_dataset, filtered_indices)

    target_map = {
        base_dataset.class_to_idx[class_name]: index for index, class_name in enumerate(ALLOWED_CLASSES)
    }
    return FilteredImageFolderSubset(
        base_dataset=base_dataset,
        indices=filtered_indices,
        target_map=target_map,
        transform=transform,
    )


# Build the train, validation, and test datasets directly from the generated dataset folders.
def build_datasets(
    data_dir: str | Path = "dataset",
    verify_images: bool = True,
) -> Tuple[Dict[str, FilteredImageFolderSubset], Dict[str, int]]:
    transforms = get_transforms()
    datasets = {
        split_name: build_split_dataset(
            data_dir=data_dir,
            split_name=split_name,
            transform=transforms[split_name],
            verify_images=verify_images,
        )
        for split_name in SPLIT_ORDER
    }

    class_to_idx = {class_name: index for index, class_name in enumerate(ALLOWED_CLASSES)}
    return datasets, class_to_idx


# Wrap each prepared split dataset in a DataLoader with deterministic training shuffling.
def create_dataloaders(
    data_dir: str | Path = "dataset",
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = DEFAULT_SEED,
    verify_images: bool = True,
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    datasets, class_to_idx = build_datasets(data_dir=data_dir, verify_images=verify_images)
    generator = torch.Generator().manual_seed(seed)
    dataloaders = {
        split_name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            generator=generator if split_name == "train" else None,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
        )
        for split_name, dataset in datasets.items()
    }
    return dataloaders, class_to_idx


# Render a compact split summary for quick manual verification from the command line.
def summarize_splits(datasets: Dict[str, FilteredImageFolderSubset]) -> str:
    lines = []
    for split_name, dataset in datasets.items():
        lines.append(f"{split_name}: {len(dataset)} images")
    return "\n".join(lines)


# Provide a lightweight CLI check that the generated dataset layout can be loaded successfully.
def main() -> None:
    datasets, class_to_idx = build_datasets()
    print("Prepared ImageFolder datasets for EfficientNet-B0:")
    print(summarize_splits(datasets))
    print(f"class_to_idx: {class_to_idx}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B0_Weights
from torchvision.transforms import InterpolationMode, v2

ALLOWED_CLASSES: Tuple[str, ...] = ("normal", "blast", "brown_spot")
SPLIT_RATIOS: Dict[str, float] = {"train": 0.6, "val": 0.2, "test": 0.2}
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 0
DEFAULT_SEED = 42
SPLIT_ORDER: Tuple[str, ...] = ("train", "val", "test")


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


def get_transforms() -> Dict[str, v2.Compose]:
    weights = EfficientNet_B0_Weights.DEFAULT
    pretrained_transforms = weights.transforms()
    normalize = v2.Normalize(mean=pretrained_transforms.mean, std=pretrained_transforms.std)

    train_transform = v2.Compose(
        [
            v2.RandomResizedCrop(
                size=224,
                scale=(0.9, 1.0),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            normalize,
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.Resize(size=256, interpolation=InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(size=224),
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=True),
            normalize,
        ]
    )

    return {"train": train_transform, "val": eval_transform, "test": eval_transform}


def load_base_dataset(data_dir: str | Path = "data/train_images") -> ImageFolder:
    dataset = ImageFolder(root=str(data_dir))
    missing_classes = [class_name for class_name in ALLOWED_CLASSES if class_name not in dataset.class_to_idx]
    if missing_classes:
        missing = ", ".join(sorted(missing_classes))
        raise ValueError(f"Missing required class folders in {data_dir}: {missing}")

    return dataset


def filter_indices(dataset: ImageFolder, class_names: Iterable[str] = ALLOWED_CLASSES) -> List[int]:
    allowed_target_ids = {dataset.class_to_idx[class_name] for class_name in class_names}
    filtered_indices = [
        index for index, target in enumerate(dataset.targets) if target in allowed_target_ids
    ]

    if not filtered_indices:
        allowed = ", ".join(class_names)
        raise ValueError(f"No images found for the requested classes: {allowed}")

    return filtered_indices


def _split_counts(total_items: int) -> Tuple[int, int, int]:
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


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


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


def split_indices(
    dataset: ImageFolder,
    filtered_indices: Sequence[int],
    seed: int = DEFAULT_SEED,
) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    grouped_indices: Dict[int, List[int]] = defaultdict(list)

    for index in filtered_indices:
        _, target = dataset.samples[index]
        grouped_indices[target].append(index)

    split_map: Dict[str, List[int]] = {"train": [], "val": [], "test": []}

    for target, indices in grouped_indices.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        if len(shuffled) < len(SPLIT_ORDER):
            class_name = dataset.classes[target]
            raise ValueError(
                f"Class '{class_name}' has only {len(shuffled)} image(s). "
                "At least 3 are required to keep train, val, and test non-empty."
            )
        train_count, val_count, _ = _split_counts(len(shuffled))

        split_map["train"].extend(shuffled[:train_count])
        split_map["val"].extend(shuffled[train_count : train_count + val_count])
        split_map["test"].extend(shuffled[train_count + val_count :])

    return split_map


def build_datasets(
    data_dir: str | Path = "data/train_images",
    seed: int = DEFAULT_SEED,
    verify_images: bool = True,
) -> Tuple[Dict[str, FilteredImageFolderSubset], Dict[str, int]]:
    base_dataset = load_base_dataset(data_dir)
    transforms = get_transforms()
    filtered_indices = filter_indices(base_dataset)
    if verify_images:
        validate_class_images(base_dataset, filtered_indices)
    split_map = split_indices(base_dataset, filtered_indices, seed=seed)
    target_map = {
        base_dataset.class_to_idx[class_name]: index for index, class_name in enumerate(ALLOWED_CLASSES)
    }

    datasets = {
        split_name: FilteredImageFolderSubset(
            base_dataset=base_dataset,
            indices=indices,
            target_map=target_map,
            transform=transforms[split_name],
        )
        for split_name, indices in split_map.items()
    }

    class_to_idx = {class_name: index for index, class_name in enumerate(ALLOWED_CLASSES)}
    return datasets, class_to_idx


def create_dataloaders(
    data_dir: str | Path = "data/train_images",
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    seed: int = DEFAULT_SEED,
    verify_images: bool = True,
) -> Tuple[Dict[str, DataLoader], Dict[str, int]]:
    datasets, class_to_idx = build_datasets(data_dir=data_dir, seed=seed, verify_images=verify_images)
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


def summarize_splits(datasets: Dict[str, FilteredImageFolderSubset]) -> str:
    lines = []
    for split_name, dataset in datasets.items():
        lines.append(f"{split_name}: {len(dataset)} images")
    return "\n".join(lines)


if __name__ == "__main__":
    datasets, class_to_idx = build_datasets()
    print("Prepared ImageFolder subsets for EfficientNet-B0:")
    print(summarize_splits(datasets))
    print(f"class_to_idx: {class_to_idx}")

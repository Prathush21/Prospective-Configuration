import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union, List
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torch
from torchvision import transforms


class CustomDataset(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. Ensure the dataset is in the correct directory."
            )

        self.classes, self.class_to_idx = self._find_classes(
            os.path.join(self.root, "train")
        )

        if self.train:
            self.data, self.targets = self._load_data(os.path.join(self.root, "train"))
        else:
            self.data, self.targets = self._load_data(os.path.join(self.root, "val"))

        if self.transform is None:
            self.transform = transforms.Compose(
                [transforms.Resize((32, 32)), transforms.ToTensor()]
            )

    def _find_classes(self, dir: str) -> Tuple[Any, Any]:
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _load_data(self, dir: str) -> Tuple[List[np.ndarray], List[int]]:
        # Implement your logic to load data here
        # This example assumes images are stored in class-labeled subdirectories
        data = []
        targets = []
        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(dir, target_class)
            for root, _, fnames in sorted(os.walk(target_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    try:
                        img = Image.open(path).convert(
                            "RGB"
                        )  # Ensure image is in RGB format
                        img = np.array(img, dtype=np.uint8)  # Ensure image is uint8
                        data.append(img)
                        targets.append(class_index)
                    except Exception as e:
                        print(f"Error loading image {path}: {e}")
        return data, targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        img, target = self.data[index], self.targets[index]

        # Convert numpy array to PIL Image
        try:
            img = Image.fromarray(img)
        except Exception as e:
            print(f"Error converting numpy array to image at index {index}: {e}")
            print(f"Array shape: {img.shape}, dtype: {img.dtype}")
            raise

        if self.transform is not None:
            print("Applying transform...")

            img = self.transform(img)
        if self.target_transform is not None:
            print("Applying target transform...")

            target = self.target_transform(target)

        return img, target

    def _check_integrity(self) -> bool:
        train_dir = os.path.join(self.root, "train")
        val_dir = os.path.join(self.root, "val")
        return os.path.isdir(train_dir) and os.path.isdir(val_dir)

    def extra_repr(self) -> str:
        split = "Train" if self.train else "Val"
        return f"Split: {split}"

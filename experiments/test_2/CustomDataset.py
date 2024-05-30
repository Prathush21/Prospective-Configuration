import os
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torch
from torchvision import transforms


class CustomDataset(VisionDataset):
    def init(
        self,
        root: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().init(root, transform=transform, target_transform=target_transform)
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

    def _find_classes(self, dir: str) -> Tuple[Any, Any]:
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _load_data(self, dir: str) -> Tuple[np.ndarray, list]:
        data = []
        targets = []
        for class_name in sorted(os.listdir(dir)):
            if class_name not in self.class_to_idx:
                continue
            class_index = self.class_to_idx[class_name]
            class_dir = os.path.join(dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for root, _, fnames in sorted(os.walk(class_dir)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    image = Image.open(path).convert("RGB")
                    if self.transform is not None:
                        image = self.transform(image)
                    data.append(np.array(image))
                    targets.append(class_index)
        data = np.stack(data)
        return data, targets

    def len(self) -> int:
        return len(self.data)

    def getitem(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _check_integrity(self) -> bool:
        train_dir = os.path.join(self.root, "train")
        val_dir = os.path.join(self.root, "val")
        return os.path.isdir(train_dir) and os.path.isdir(val_dir)

    def extra_repr(self) -> str:
        split = "Train" if self.train else "Val"
        return f"Split: {split}"

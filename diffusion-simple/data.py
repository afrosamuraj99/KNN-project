from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader, default_collate
import torchvision.transforms.functional as F
from PIL import Image


class DataTransform:
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, data):
        if isinstance(data, list):
            return default_collate([self.transform_one(x) for x in data])
        else:
            return self.transform_one(data)

    def transform_one(self, data):
        img = Image.open(data["img_path"]).convert("RGB")

        if self.transform is not None:
            data["img"] = self.transform(img)
            # Create grayscale version (average across RGB channels)
            data["grayscale"] = data["img"].mean(dim=0, keepdim=True)  # Simple grayscale conversion
        else:
            data["img"] = F.pil_to_tensor(img)
            data["grayscale"] = data["img"].mean(dim=0, keepdim=True)
        
        return data


class DatasetLister:
    def __init__(self, path: str):
        self.path = Path(path)
        self.img_paths = []

        for img_path in self.path.iterdir():
            self.img_paths.append(str(img_path))

    def __getitem__(self, i: int) -> dict:
        data = {"img_path": self.img_paths[i]}
        return data

    def __len__(self):
        return len(self.img_paths)

    def __iter__(self) -> Iterator[dict]:
        for i in range(len(self.labels)):
            yield {"img_path": self.img_paths[i]}


class Dataset(TorchDataset):
    def __init__(self, path: str, transform=None):
        self.data = DatasetLister(path)
        self.tx = DataTransform(transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        return self.tx(self.data[i])


def setup_loader(*, data_dir, batch_size, shuffle, transform=None):
    dataset = Dataset(data_dir, transform)
    loader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader

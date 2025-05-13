from pathlib import Path
from typing import Any, Iterator

from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader, default_collate
import torchvision.transforms.functional as F
from torchvision import transforms as T
from PIL import Image


class DataTransform:
    def __init__(self, transforms):
        self.img_tx = transforms["img"]
        self.ref_tx = transforms["ref"]

    def __call__(self, data):
        if isinstance(data, list):
            return default_collate([self.transform_one(x) for x in data])
        else:
            return self.transform_one(data)

    def transform_one(self, data):
        img = Image.open(data["img_path"]).convert("RGB")
        ref = Image.open(data["ref_path"]).convert("RGB")

        data["img"] = self.img_tx(img)
        data["grayscale"] = data["img"].mean(dim=0, keepdim=True)
        data["ref"] = self.ref_tx(ref)

        return data


class DatasetLister:
    def __init__(self, path: str):
        self.path = Path(path)
        self.references_path = self.path.parent / (self.path.name + "_references")

        self.img_paths = []
        self.ref_paths = []

        for img_path in self.path.iterdir():
            self.img_paths.append(str(img_path))
            self.ref_paths.append(str(self.references_path / img_path.name))

    def __getitem__(self, i: int) -> dict:
        data = {
            "img_path": self.img_paths[i],
            "ref_path": self.ref_paths[i],
        }
        return data

    def __len__(self):
        return len(self.img_paths)

    def __iter__(self) -> Iterator[dict]:
        for i in range(len(self.labels)):
            yield {
                "img_path": self.img_paths[i],
                "ref_path": self.ref_paths[i],
            }


class Dataset(TorchDataset):
    def __init__(self, path: str, transforms):
        self.data = DatasetLister(path)
        self.tx = DataTransform(transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        return self.tx(self.data[i])


def setup_loader(*, data_dir, batch_size, shuffle, transforms):
    dataset = Dataset(data_dir, transforms)
    loader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader

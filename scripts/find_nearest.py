from pathlib import Path
import argparse
import os
import shutil

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader, default_collate
import torchvision.transforms.functional as F
from torchvision import transforms as T
import clip
from tqdm import tqdm
from PIL import Image


class DataTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data):
        if isinstance(data, list):
            return default_collate([self.transform_one(x) for x in data])
        else:
            return self.transform_one(data)

    def transform_one(self, data):
        img = Image.open(data["img_path"]).convert("RGB")
        data["img"] = self.transform(img)
        return data


class DatasetLister:
    def __init__(self, path: str):
        self.path = Path(path)
        self.img_paths = []
        for img_path in self.path.iterdir():
            self.img_paths.append(str(img_path))

    def __getitem__(self, i: int) -> dict:
        data = {
            "img_path": self.img_paths[i],
        }
        return data

    def __len__(self):
        return len(self.img_paths)

    def __iter__(self):
        for i in range(len(self.img_paths)):
            yield {
                "img_path": self.img_paths[i],
            }


class Dataset(TorchDataset):
    def __init__(self, path: str, transform):
        self.data = DatasetLister(path)
        self.tx = DataTransform(transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        return self.tx(self.data[i])


def setup_loader(*, data_dir, batch_size, shuffle, transform):
    dataset = Dataset(data_dir, transform)
    loader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader


if __name__ == "__main__":
    assert len(os.environ["CUDA_VISIBLE_DEVICES"]) == 1

    device = (
        torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="path to the dataset directory", required=True)
    ap.add_argument("--resolution", help="dataset resolution", required=True, type=int)
    ap.add_argument("--split", help="dataset split", required=True)
    ap.add_argument("--batch-size", type=int, default="256")
    args = ap.parse_args()

    model, preprocess = clip.load("RN50", device=device)

    image_size = args.resolution
    batch_size = args.batch_size
    dataset_path_str = args.dataset
    split = args.split

    dset_path = Path(dataset_path_str) / f"{split}_imgs" / f"resized_{image_size}"
    references_path = dset_path.with_name(dset_path.name + "_references")
    references_path.mkdir(exist_ok=True)

    train_dataloader = setup_loader(
        data_dir=str(dset_path),
        batch_size=batch_size,
        shuffle=False,
        transform=preprocess,
    )

    model.eval()
    with torch.no_grad():
        image_features_all = []
        paths_all = []

        for batch in tqdm(train_dataloader):
            images = batch["img"].to(device)
            image_features = model.encode_image(images)
            image_features_all.append(image_features)

            paths = batch["img_path"]
            paths_all.extend(paths)

    features = torch.cat(image_features_all, dim=0)
    norm = features.norm(p=2, dim=-1, keepdim=True)
    features /= norm

    indices = []
    for feats in features:
        _vals, idxs = (feats @ features.T).topk(k=2, dim=-1)
        indices.append(idxs.cpu())

    indices = torch.stack(indices, dim=0)
    # Assumption:
    # The 1st most similar to an image is itself, so pick the 2nd.
    # But sometimes that's not the case, then pick the 1st.
    assumption_holds = indices[:, 0] == torch.arange(features.size(0))
    neighbors = torch.where(assumption_holds, indices[:, 1], indices[:, 0])
 
    for image_idx, neighbor_idx in enumerate(neighbors):
        dst_path = references_path / Path(paths_all[image_idx]).name
        shutil.copyfile(paths_all[neighbor_idx], dst_path)

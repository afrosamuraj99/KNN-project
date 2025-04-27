import argparse
import pickle
import os
from pathlib import Path

import numpy as np
import torch
import torchvision

from model import load, load_ema
from utils.misc import num_to_groups, latest_checkpoint, epoch_checkpoint
from diffusion import sample


def old_load(path, mode):
    checkpoint = torch.load(path, weights_only=True, mmap=True, map_location=torch.device('cpu'))

    with torch.device("meta"):
        model = Unet(
            channels=checkpoint["channels"],
            init_dim=checkpoint["image_size"],
            dim_mults=checkpoint["dim_mults"],
        )
    model.load_state_dict(checkpoint["model_state_dict"], assign=True)

    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
    else:
        RuntimeError("Supported modes are 'eval' or 'train'")

    return model, optimizer

def old_do(epoch):
    model, opt = old_load(f"./out/basic/checkpoints/epoch-{epoch}.pth", "train")
    batches = num_to_groups(1, 1)
    all_images_list = list(map(lambda n: sample(model, image_size=128, batch_size=n, channels=3), batches))
    all_images = torch.cat(all_images_list, dim=0)
    all_images = (all_images + 1) / 2
    torchvision.utils.save_image(all_images, f"sample-{epoch}.png", nrow=8)

def load_history(path_str, print=True):
    with open(path_str, "rb") as f:
        hist = pickle.load(f)
    if print:
        for k, v in hist.items():
            a = np.array(v)
            if len(a):
                print(k, np.max(v))
    return hist

def do(model, out):
    batches = num_to_groups(16, 16)
    all_images_list = list(map(lambda n: sample(model, image_size=128, batch_size=n, channels=3), batches))
    all_images = torch.cat(all_images_list, dim=0)
    all_images = (all_images + 1) / 2
    torchvision.utils.save_image(all_images, out, nrow=8)


if __name__ == "__main__":
    assert len(os.environ["CUDA_VISIBLE_DEVICES"]) == 1

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--epoch", required=False, type=int, default=-1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.epoch == -1:
        ckpt, ema_ckpt = latest_checkpoint(Path("./out") / args.name / "checkpoints")
    else:
        ckpt, ema_ckpt = epoch_checkpoint(Path("./out") / args.name / "checkpoints", args.epoch)

    if args.ema:
        print(f"Loading: {ema_ckpt}")
        model = load_ema(ema_ckpt, "eval")
    else:
        print(f"Loading: {ckpt}")
        model, _ = load(ckpt, "eval")

    device = (
        torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
    )

    model.to(device)

    print(f"Sampling")
    do(model, args.out)

import argparse
import os
from pathlib import Path

import torch
import torchvision

from model import load_model, load_ema
from utils.misc import num_to_groups, latest_checkpoint, epoch_checkpoint
from diffusion import sample
from scheduling import Schedule, ScheduleDDIM, load_schedule_kwargs, linear_beta_schedule


def load_model_v2(path, mode):
    from model import Unet
    checkpoint = torch.load(path, weights_only=True, mmap=True)
    with torch.device("meta"):
        model = Unet(
            channels=checkpoint["channels"],
            init_dim=checkpoint["init_dim"],
            dim_mults=checkpoint["dim_mults"],
        )
    model.load_state_dict(checkpoint["model_state_dict"], assign=True)
    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
    else:
        RuntimeError("Supported modes are 'eval' or 'train'")
    return model

def load_ema_v2(path, mode):
    from model import Unet
    checkpoint = torch.load(path, weights_only=True, mmap=True)
    with torch.device("meta"):
        model = Unet(
            channels=checkpoint["channels"],
            init_dim=checkpoint["init_dim"],
            dim_mults=checkpoint["dim_mults"],
        )
        ema_model = torch.optim.swa_utils.AveragedModel(
            model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(checkpoint["ema_decay"])
        )
    ema_model.load_state_dict(checkpoint["model_state_dict"], assign=True)
    if mode == "eval":
        ema_model.eval()
    elif mode == "train":
        ema_model.train()
    else:
        RuntimeError("Supported modes are 'eval' or 'train'")
    return ema_model

def get_sched_kwargs_v2():
    sched_kwargs = {"betas": linear_beta_schedule(timesteps=1000)}
    return sched_kwargs

def do(model, out, num_samples, batch_size, sched):
    batches = num_to_groups(num_samples, batch_size)
    all_images_list = list(map(lambda n: sample(model, sched=sched, image_size=128, batch_size=n, channels=3), batches))
    all_images = torch.cat(all_images_list, dim=0)
    all_images = (all_images + 1) / 2
    torchvision.utils.save_image(all_images, out, nrow=8)


if __name__ == "__main__":
    assert len(os.environ["CUDA_VISIBLE_DEVICES"]) == 1

    ap = argparse.ArgumentParser()

    group1 = ap.add_mutually_exclusive_group(required=True)
    group1.add_argument("--name")
    group1.add_argument("--checkpoint")

    group2 = ap.add_mutually_exclusive_group(required=True)
    group2.add_argument("--ddpm", action="store_true")
    group2.add_argument("--ddim", action="store_true")

    ap.add_argument("--out", required=True)
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--epoch", required=False, type=int, default=-1)
    ap.add_argument("--ddim_steps", required=False, type=int, default=25)
    ap.add_argument("--num_samples", required=False, type=int, default=16)
    ap.add_argument("--batch_size", required=False, type=int, default=16)

    args = ap.parse_args()

    if args.checkpoint is None:
        experiment_path = Path("./out") / args.name
        ckpts_path = experiment_path / "checkpoints"
        schedule_path = experiment_path / "schedule_kwargs.pt"

        if args.epoch == -1:
            ckpt, ema_ckpt, _opt_ckpt = latest_checkpoint(ckpts_path)
        else:
            ckpt, ema_ckpt, _opt_ckpt = epoch_checkpoint(ckpts_path, args.epoch)

        print(f"Loading schedule: {schedule_path}")
        sched_kwargs = load_schedule_kwargs(schedule_path)
        sched = Schedule(**sched_kwargs)
        ddim_sched = ScheduleDDIM(args.ddim_steps, sched, **sched_kwargs)
    else:
        lut = {
            "2": (load_model_v2, load_ema_v2, get_sched_kwargs_v2),
            "3": (load_model, load_ema)
        }
        ckpt, version = args.checkpoint.split(",")

        if version not in lut:
            assert RuntimeError(f"Unsupported version {version}")

        load_model, load_ema, get_sched_kwargs = lut[version]

        if args.ema and not load_ema:
            assert RuntimeError(f"Lazy or impossible to implement ema sampling on version {version}")

        sched_kwargs = get_sched_kwargs()
        sched = Schedule(**sched_kwargs)
        ddim_sched = ScheduleDDIM(args.ddim_steps, sched, **sched_kwargs)

    if args.ema:
        load_model = load_ema

    print(f"Loading model: {ckpt}")
    model = load_model(ckpt, "eval")

    device = (
        torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
    )
    model.to(device)

    print(f"Sampling")
    do(model, args.out, args.num_samples, args.batch_size, ddim_sched if args.ddim else sched)

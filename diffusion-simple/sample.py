import random
import argparse
import os
from pathlib import Path
from functools import partial

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from tqdm import tqdm

from model import load_model, load_ema
from utils.misc import num_to_groups, latest_checkpoint, epoch_checkpoint
from diffusion import sample, predict_start_from_noise
from scheduling import Schedule, ScheduleDDIM, load_schedule_kwargs, linear_beta_schedule, extract


USE_ALT_DDIM_IMPL = False
picked_sample_fn = None


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


def do(model, out, num_samples, batch_size, sched, grayscale_path=None):
    global picked_sample_fn

    batches = num_to_groups(num_samples, batch_size)
    device = next(model.parameters()).device

    grayscale_cond = None
    if grayscale_path:
        dataset = list(Path(grayscale_path).glob("*.png")) + list(Path(grayscale_path).glob("*.jpg"))
        if len(dataset) == 0:
            raise ValueError("Dataset path does not contain any valid images.")

        # Select 16 unique random images
        selected_images = random.sample(dataset, min(len(dataset), num_samples))

        # Process all selected images
        grayscale_tensors = []
        color_images = []
        for img_path in selected_images:
            color_img = Image.open(img_path).convert("RGB")
            color_img = color_img.resize((32, 32))
            color_tensor = F.to_tensor(color_img)
            color_images.append(color_tensor)

            grayscale_tensor = color_tensor.mean(dim=0, keepdim=True)
            grayscale_tensor = (grayscale_tensor * 2) - 1
            grayscale_tensors.append(grayscale_tensor)

        grayscale_cond = torch.stack(grayscale_tensors).to(device)
        color_display = torch.stack(color_images).to(device)  # Přesun na stejné zařízení jako model

        # Save original color images (top half)
        color_display_save = (color_display * 2) - 1
        torchvision.utils.save_image(color_display_save, out, nrow=8)
    else:
        grayscale_cond = torch.rand((batch_size, 1, 32, 32), device=device) * 2 - 1

    # Generate colorized versions (bottom half)
    all_images_list = list(map(
        lambda n: picked_sample_fn(
            model,
            sched=sched,
            image_size=32,
            batch_size=n,
            channels=3,
            grayscale=grayscale_cond[:n] if grayscale_cond is not None else None
        ),
        batches
    ))

    all_images = torch.cat(all_images_list, dim=0)
    all_images = (all_images + 1) / 2

    # Combine original and colorized images
    if grayscale_path:
        combined_images = torch.cat([
            color_display,  # Original color images (top)
            all_images      # Colorized images (bottom)
        ])
        torchvision.utils.save_image(combined_images, out, nrow=8)
    else:
        torchvision.utils.save_image(all_images, out, nrow=8)


@torch.inference_mode()
def alt_ddim_sample_wrapper(model, *, image_size, batch_size, channels, sched, grayscale):
    device = next(model.parameters()).device
    shape = (batch_size, channels, image_size, image_size)
    return alt_ddim_sample(model, shape, device, sched.base.timesteps, sched.timesteps, 0.0, grayscale, sched)


@torch.inference_mode()
def alt_ddim_sample(model, shape, device, total_timesteps, sampling_timesteps, eta, grayscale, sched, return_all_timesteps=False):
    times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    img = torch.randn(shape, device=device)
    # imgs = [img]

    x_start = None

    for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
        time_cond = torch.full((shape[0],), time, device = device, dtype = torch.long)
        time_next_cond = torch.full((shape[0],), time_next, device = device, dtype = torch.long)

        pred_noise = model(img, time_cond, grayscale=grayscale)
        x_start = predict_start_from_noise(img, time_cond, pred_noise, sched.base)
        x_start = torch.clamp(x_start, min=-1., max=1.)

        if time_next < 0:
            img = x_start
            # imgs.append(img)
            continue

        alpha = sched.base.alphas_cumprod[time]
        alpha_next = sched.base.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(img)

        img = x_start * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise

        # imgs.append(img)

    # ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)
    ret = img
    return ret


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
    ap.add_argument("--base_path", default="./out")
    ap.add_argument("--seed", default=None)
    ap.add_argument("--epoch", required=False, type=int, default=-1)
    ap.add_argument("--ddim_steps", required=False, type=int, default=25)
    ap.add_argument("--num_samples", required=False, type=int, default=16)
    ap.add_argument("--batch_size", required=False, type=int, default=16)
    ap.add_argument("--grayscale", required=False, help="Path to grayscale image for conditioning", default="../diffusion/datasets/cifar_test")

    args = ap.parse_args()
    base_path = Path(args.base_path)
    seed = args.seed

    if seed is not None:
        random.seed(int(seed))
        torch.manual_seed(int(seed))

    picked_sample_fn = sample
    if args.ddim is True:
        if USE_ALT_DDIM_IMPL is True:
            picked_sample_fn = alt_ddim_sample_wrapper

    if args.checkpoint is None:
        experiment_path = base_path / args.name
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
        ckpt, version = args.checkpoint.split(",")

        lut = {
            "2": (load_model_v2, load_ema_v2, get_sched_kwargs_v2),
            "3": (load_model, load_ema, partial(load_schedule_kwargs, Path(ckpt).parent.parent / "schedule_kwargs.pt"))
        }

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
    do(model, args.out, args.num_samples, args.batch_size, 
        ddim_sched if args.ddim else sched,
        grayscale_path=args.grayscale)

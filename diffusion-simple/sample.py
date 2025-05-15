import random
import argparse
import os
from pathlib import Path
from functools import partial

import torch
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms as T
import clip
from PIL import Image
from tqdm import tqdm

from model import load_model, load_ema, Unet as BigUnet
from small_model import Unet as SmallUnet
from utils.misc import num_to_groups, latest_checkpoint, epoch_checkpoint
from diffusion import sample, predict_start_from_noise, extract_clip_features, hook_clip
from scheduling import Schedule, ScheduleDDIM, load_schedule_kwargs, linear_beta_schedule, extract


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


def do(model, device, image_size, out, num_samples, batch_size, sched, images_path, with_reference):
    global picked_sample_fn

    out = Path(out)
    img_base_path = Path(images_path)
    ref_base_path = img_base_path.with_name(img_base_path.name + "_references")

    cpu = torch.device("cpu")
    model.to(device)
    model.eval()
    if with_reference is True:
        resnet_full, _resnet_preprocess = clip.load("RN50", device="cpu")
        clip_model = resnet_full.visual
        clip_model.eval()
        clip_model.to(device)
        _clip_hooks = hook_clip(clip_model)
        ref_transform = T.Compose([
            T.Resize(clip_model.input_resolution, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(clip_model.input_resolution),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    # Select 16 unique random images
    # selected_images = random.sample(dataset, min(len(dataset), num_samples))
    selected_images = []
    for idx, img_path in enumerate(img_base_path.iterdir()):
        if num_samples != -1 and idx == num_samples:
            break
        selected_images.append(img_path.name)

    if len(selected_images) == 0:
        raise ValueError("Colored path does not contain any images.")

    if num_samples != -1:
        batches = num_to_groups(num_samples, batch_size)
        assert len(batches) == 1
    else:
        batches = num_to_groups(len(selected_images), batch_size)

    # Process all selected images
    finished = 0
    for n in batches:
        colored_images = []
        grayscale_images = []
        reference_images = []
        image_names = selected_images[finished : finished + n]

        for img_name in image_names:
            img_path = img_base_path / img_name
            color_img = Image.open(img_path).convert("RGB")
            color_tensor = F.to_tensor(color_img)
            colored_images.append(color_tensor)

            grayscale_tensor = color_tensor.mean(dim=0, keepdim=True)
            grayscale_images.append(grayscale_tensor)

            if with_reference is True:
                img_path = ref_base_path / img_name
                ref_img = Image.open(img_path).convert("RGB")
                ref_tensor = F.to_tensor(ref_img)
                reference_images.append(ref_tensor)

        grayscale_cond = ((torch.stack(grayscale_images) * 2) - 1).to(device)

        if with_reference is True:
            ref_cond = torch.stack(reference_images).to(device)
            clip_input = ref_transform(ref_cond)
            clip_features = extract_clip_features(clip_model, clip_input)

        # Generate colorized versions (bottom half)
        generated = picked_sample_fn(
            model,
            sched=sched,
            image_size=image_size,
            batch_size=n,
            channels=3,
            grayscale=grayscale_cond,
            clip_features=clip_features if with_reference is True else None,
        )
        generated_display = (generated + 1) / 2
        generated_display = generated_display.to(cpu)

        if num_samples != -1:
            to_cat = []
            for idx in range(num_samples):
                if with_reference is True:
                    ref_display = F.resize(reference_images[idx], (image_size, image_size))
                    to_cat.append(ref_display.unsqueeze(0))
                to_cat.append(colored_images[idx].unsqueeze(0))
                to_cat.append(grayscale_images[idx].repeat(3, 1, 1).unsqueeze(0))
                to_cat.append(generated_display[idx].unsqueeze(0))
            combined_images = torch.cat(to_cat, dim=0)
            torchvision.utils.save_image(combined_images, out, nrow=4 if with_reference is True else 3)
        else:
            for name, generated in zip(image_names, generated_display):
                torchvision.utils.save_image(generated, out / name)
        finished += n


@torch.inference_mode()
def alt_ddim_sample_wrapper(model, *, image_size, batch_size, channels, sched, grayscale, clip_features):
    device = next(model.parameters()).device
    shape = (batch_size, channels, image_size, image_size)
    return alt_ddim_sample(model, shape, device, sched.base.timesteps, sched.timesteps, 0.0, grayscale, clip_features, sched)


@torch.inference_mode()
def alt_ddim_sample(model, shape, device, total_timesteps, sampling_timesteps, eta, grayscale, clip_features, sched, return_all_timesteps=False):
    times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    img = torch.randn(shape, device=device)
    # imgs = [img]

    x_start = None

    for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
        time_cond = torch.full((shape[0],), time, device = device, dtype = torch.long)
        time_next_cond = torch.full((shape[0],), time_next, device = device, dtype = torch.long)

        pred_noise = model(img, time_cond, grayscale=grayscale, clip_features=clip_features)
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
    group2.add_argument("--altddim", action="store_true")

    group3 = ap.add_mutually_exclusive_group(required=True)
    group3.add_argument("--small", help="is a simpler model", action="store_true")
    group3.add_argument("--big", help="is a complicated model", action="store_true")

    ap.add_argument("--colored", required=True, help="Path to a colored dataset, will be converted to grayscale")
    ap.add_argument("--resolution", required=True, help="Resolution of --colored images", type=int)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_samples", required=True, type=int)
    ap.add_argument("--batch_size", required=True, type=int)
    ap.add_argument("--reference", action="store_true", help="Wheter to also include a reference images to the --colored ones")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--base_path", default="out")
    ap.add_argument("--seed", default=None)
    ap.add_argument("--epoch", required=False, type=int, default=-1)
    ap.add_argument("--ddim_steps", required=False, type=int, default=25)

    args = ap.parse_args()

    if args.num_samples == -1:
        out_dir_path = Path(args.out)
        # Do not overwrite anything
        assert not out_dir_path.exists()
        out_dir_path.mkdir(exist_ok=False, parents=False)
    else:
        out_path = Path(args.out)
        if out_path.exists():
            # Do not overwrite directories
            assert out_path.is_file()

    seed = args.seed
    if seed is not None:
        random.seed(int(seed))
        torch.manual_seed(int(seed))

    picked_sample_fn = sample
    if args.altddim is True:
        args.ddim = True
        picked_sample_fn = alt_ddim_sample_wrapper

    base_path = Path(args.base_path)
    if args.checkpoint is None:
        experiment_path = base_path / args.name
        ckpts_path = experiment_path / "checkpoints"
        schedule_path = experiment_path / "schedule_kwargs.pt"

        if args.epoch == -1:
            ckpt, ema_ckpt, _opt_ckpt = latest_checkpoint(ckpts_path)
        else:
            ckpt, ema_ckpt, _opt_ckpt = epoch_checkpoint(ckpts_path, args.epoch)

        if args.ema:
            ckpt = ema_ckpt

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
    if args.small is True:
        Model = SmallUnet
    elif args.big is True:
        Model = BigUnet
    else:
        raise RuntimeError("Should be unreachable")
    model = load_model(ckpt, Model, "eval")

    device = (
        torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
    )

    print(f"Sampling")
    do(
        model, device, args.resolution, args.out, args.num_samples, args.batch_size,
        ddim_sched if args.ddim else sched,
        images_path=args.colored, with_reference=args.reference,
    )

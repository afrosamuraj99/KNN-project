import argparse
import math
from pathlib import Path
from collections import deque
import pickle

import torch
import torch.nn.functional as F
import torchvision
from tqdm.auto import tqdm

from data import setup_loader
from utils.misc import (
    exists, default, num_to_groups, transform,
    reverse_transform, latest_checkpoint,
)
from scheduling import linear_beta_schedule, extract
from model import load, save, Unet


timesteps = 300

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def get_noisy_image(x_start, t):
  # add noise
  x_noisy = q_sample(x_start, t=t)

  # turn back into PIL image
  reverse_image = reverse_transform(x_noisy.squeeze())
  noisy_image = torchvision.transforms.functional.to_pil_image(reverse_image, mode=None)

  return noisy_image

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l2"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    imgs = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(0, timesteps)), desc="Sampling loop time step", total=timesteps, unit="step", leave=False):
        imgs = p_sample(model, imgs, torch.full((b,), i, device=device, dtype=torch.long), i)
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="experiment name", required=True)
    ap.add_argument("--dataset", help="path to the dataset directory", required=True)
    ap.add_argument("--resolution", help="dataset resolution", type=int, default=128)
    args = ap.parse_args()

    name = args.name
    image_size = args.resolution
    channels = 3
    dim_mults = (1, 2, 4,)
    epochs = 10
    batch_size = 25
    save_and_sample_every = 500  # nth batch

    train_dset_path = Path(args.dataset) / "train_imgs" / f"resized_{image_size}"
    val_dset_path = Path(args.dataset) / "val_imgs" / f"resized_{image_size}"

    out_path = Path("./out") / name
    checkpoints_folder = out_path / "checkpoints"
    results_folder = out_path / "results"
    history_path = out_path / "history.pkl"

    checkpoints_folder.mkdir(exist_ok=True)
    results_folder.mkdir(exist_ok=True)

    train_dataloader = setup_loader(
        data_dir=str(train_dset_path),
        batch_size=batch_size,
        shuffle=True,
        transform=transform,
    )
    nr_batches = math.ceil(len(train_dataloader.dataset) / batch_size)

    device = (
        torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
    )

    checkpoint_path = latest_checkpoint(checkpoints_folder)
    if checkpoint_path is None:
        print("Creating new model")
        model = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=dim_mults,
        )
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        start_epoch = 0
        history = {}
    else:
        with history_path.open("rb") as f:
            history = pickle.load(f)
            history_keys = list(history.keys())
        assert len(history_keys)
        start_epoch = max(history_keys) + 1
        model, optimizer = load(checkpoint_path, device, "train")
        print(f"Loaded old model, starting on epoch {start_epoch}")

    for epoch in tqdm(range(start_epoch, start_epoch + epochs), unit="epoch"):
        loss_history = deque()

        for step, batch in tqdm(enumerate(train_dataloader), unit="batch", unit_scale=True, total=nr_batches, leave=False):
            optimizer.zero_grad()

            batch_size = batch["img"].shape[0]
            batch = batch["img"].to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, batch, t, loss_type="l2")

            loss_item = loss.item()
            loss_history.append(loss_item)
            if step % 100 == 0:
                tqdm.write(f"Loss: {loss_item}")

            loss.backward()
            optimizer.step()

            # save generated images
            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                batches = num_to_groups(16, batch_size)
                all_images_list = list(map(lambda n: sample(model, image_size=image_size, batch_size=n, channels=channels), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) / 2
                torchvision.utils.save_image(all_images, str(results_folder / f"sample-{epoch}-{milestone}.png"), nrow=8)

        history[epoch] = loss_history

        # save checkpoint
        save(model, optimizer, image_size, channels, dim_mults, str(checkpoints_folder / f"epoch-{epoch}.pth"))

        with history_path.open("wb") as f:
            pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)

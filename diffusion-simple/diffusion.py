import argparse
import math
from pathlib import Path
from collections import deque
import pickle
import os

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
from model import load_model, load_ema, load_optimizer, save_model, save_ema, save_optimizer, Unet
from scheduling import Schedule, ScheduleDDIM, save_schedule_kwargs, load_schedule_kwargs


# forward diffusion (using the nice property)
def q_sample(x_start, t, sched, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sched.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sched.sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def get_noisy_image(x_start, t, sched):
  # add noise
  x_noisy = q_sample(x_start, t, sched)

  # turn back into PIL image
  reverse_image = reverse_transform(x_noisy.squeeze())
  noisy_image = torchvision.transforms.functional.to_pil_image(reverse_image, mode=None)

  return noisy_image

def p_losses(denoise_model, x_start, t, sched, noise=None, loss_type="l2"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, sched=sched, noise=noise)
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

def predict_eps_from_xstart(x_t, t, pred_xstart, sched):
    return (
        extract(sched.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
        - pred_xstart
    ) / extract(sched.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

def predict_start_from_noise(x_t, t, noise, sched):
    return (
        extract(sched.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        extract(sched.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )

def q_posterior(x_start, x_t, t, sched):
    posterior_mean = (
        extract(sched.posterior_mean_coef1, t, x_t.shape) * x_start +
        extract(sched.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance_t = extract(sched.posterior_variance, t, x_t.shape)
    return posterior_mean, posterior_variance_t

@torch.no_grad()
def p_sample(model, x, t, t_index, sched):
    # Equation 11 by using reconstructed x_0 in eq. 7.
    # Basically eq. 11 unsimplified, in order to clip intermediate x_0 into proper range.
    predicted_noise = model(x, t)
    x_recon = predict_start_from_noise(x, t, predicted_noise, sched)
    x_recon = torch.clamp(x_recon, min=-1., max=1.)
    model_mean, posterior_variance_t = q_posterior(x_start=x_recon, x_t=x, t=t, sched=sched)

    if t_index == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2
@torch.no_grad()
def p_sample_loop(model, shape, sched):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    imgs = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(0, sched.timesteps)), desc="Sampling loop time step", total=sched.timesteps, unit="step", leave=False):
        imgs = p_sample(model, imgs, torch.full((b,), i, device=device, dtype=torch.long), i, sched)
    return imgs

@torch.no_grad()
def ddim_sample(model, x, t, t_index, sched, eta=0.0, rederive_eps_from_recon=False):
    # DDPM Equation 11 by using reconstructed x_0 in eq. 7.
    # Basically eq. 11 unsimplified, in order to clip intermediate x_0 into proper range.
    eps = predicted_noise = model(x, sched.transform_times(t))
    x_recon = predict_start_from_noise(x, t, predicted_noise, sched)
    x_recon = torch.clamp(x_recon, min=-1., max=1.)
    model_mean, _posterior_variance_t = q_posterior(x_start=x_recon, x_t=x, t=t, sched=sched)

    # Should we do this if we indeed do output the eps prediction?
    # Because after clipping x_start, the rederived eps will be different.
    if rederive_eps_from_recon:
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = predict_eps_from_xstart(x, t, x_recon, sched)

    # DDIM equation 16.
    alpha_bar = extract(sched.alphas_cumprod, t, x.shape)
    alpha_bar_prev = extract(sched.alphas_cumprod_prev, t, x.shape)
    # DDPM generative process when eta = 1 and DDIM when eta = 0.
    sigma = (
        eta
        * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
        * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
    )

    # DDIM equation 12.
    mean_pred = (
        x_recon * torch.sqrt(alpha_bar_prev)
        + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
    )
    if t_index == 0:
        return mean_pred
    else:
        noise = torch.randn_like(x)
        return model_mean + sigma * noise

@torch.no_grad()
def ddim_sample_loop(model, shape, sched):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    imgs = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(0, sched.timesteps)), desc="Sampling loop time step", total=sched.timesteps, unit="step", leave=False):
        imgs = ddim_sample(model, imgs, torch.full((b,), i, device=device, dtype=torch.long), i, sched)
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3, *, sched):
    if isinstance(sched, ScheduleDDIM):
        return ddim_sample_loop(model, shape=(batch_size, channels, image_size, image_size), sched=sched)
    else:
        return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), sched=sched)

def track_samples(folder, epoch, milestone, model, microbatch_size, image_size, channels, sched):
    results_folder = folder / "results"
    results_folder.mkdir(exist_ok=True, parents=True)
    batches = num_to_groups(16, microbatch_size)
    all_images_list = list(map(lambda n: sample(model, sched=sched, image_size=image_size, batch_size=n, channels=channels), batches))
    all_images = torch.cat(all_images_list, dim=0)
    all_images = (all_images + 1) / 2
    torchvision.utils.save_image(all_images, str(folder / f"sample-{epoch}-{milestone}.png"), nrow=8)

def save_all(epoch, model, ema_model, optimizer, unet_kwargs, history, schedule_kwargs, folder):
    ckpts = folder / "checkpoints"
    ckpts.mkdir(exist_ok=True, parents=True)
    save_model(model, unet_kwargs, ckpts / f"epoch-{epoch:06d}-model.pth")
    save_ema(ema_model, ema_decay, unet_kwargs, ckpts / f"epoch-{epoch:06d}-ema.pth")
    save_optimizer(optimizer, ckpts / f"epoch-{epoch:06d}-optimizer.pth")
    with (folder / "history.pkl").open("wb") as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
    schedule_kwargs_path = folder / "schedule_kwargs.pt"
    if not schedule_kwargs_path.exists():
        save_schedule_kwargs(schedule_kwargs_path, schedule_kwargs)


if __name__ == "__main__":
    assert len(os.environ["CUDA_VISIBLE_DEVICES"]) == 1

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="experiment name", required=True)
    ap.add_argument("--dataset", help="path to the dataset directory", required=True)
    ap.add_argument("--resolution", help="dataset resolution", required=True, type=int)
    args = ap.parse_args()

    name = args.name
    image_size = args.resolution

    channels = 3
    init_dim = 128
    dim_mults = (1, 2, 2, 2,)

    ema_decay = 0.9999
    learning_rate = 1e-4
    epochs = 50
    batch_size = 256
    microbatch_size = 256
    save_and_sample_every = 250  # nth batch
    log_interval = 50

    timesteps = 1000
    ddim_timesteps = 25
    betas_f = linear_beta_schedule

    train_dset_path = Path(args.dataset) / "train_imgs" / f"resized_{image_size}"
    val_dset_path = Path(args.dataset) / "val_imgs" / f"resized_{image_size}"
    out_path = Path("./out") / name

    train_dataloader = setup_loader(
        data_dir=str(train_dset_path),
        batch_size=batch_size,
        shuffle=True,
        transform=transform,
    )
    nr_batches = math.ceil(len(train_dataloader.dataset) / batch_size)
    nr_micro_batches = math.ceil(batch_size / microbatch_size)

    device = (
        torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
    )
    unet_kwargs = {
        "channels": channels,
        "init_dim": init_dim,
        "dim_mults": dim_mults,
    }

    out_path.mkdir(exist_ok=True)
    checkpoint_path, ema_checkpoint_path, optimizer_path = latest_checkpoint(out_path / "checkpoints")

    if checkpoint_path is None:
        print("Creating new model")
        schedule_kwargs = {"betas": betas_f(timesteps=timesteps)}
        model = Unet(**unet_kwargs)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
        ema_model = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay))
        start_epoch = 1
        history = {}
    else:
        schedule_kwargs_path = out_path / "schedule_kwargs.pt"
        schedule_kwargs = load_schedule_kwargs(schedule_kwargs_path)
        history_path = out_path / "history.pkl"
        with history_path.open("rb") as f:
            history = pickle.load(f)
            history_keys = list(history.keys())
        if len(history_keys) == 0:
            start_epoch = 1
        else:
            start_epoch = max(history_keys) + 1
        model = load_model(checkpoint_path, "train")
        model.to(device)
        optimizer = load_optimizer(optimizer_path, model)
        ema_model = load_ema(ema_checkpoint_path, "train")
        print(f"Loaded old model, starting on epoch {start_epoch}")

    schedule = Schedule(**schedule_kwargs)
    ddim_schedule = ScheduleDDIM(ddim_timesteps, schedule, **schedule_kwargs)

    save_all(0, model, ema_model, optimizer, unet_kwargs, history, schedule_kwargs, out_path)
    step = 1

    for epoch in tqdm(range(start_epoch, start_epoch + epochs), unit="epoch"):
        loss_history = deque()

        for batch in tqdm(train_dataloader, unit="batch", unit_scale=True, total=nr_batches, leave=False):
            optimizer.zero_grad()

            batch_size = batch["img"].shape[0]

            mb_iter = range(0, batch_size, microbatch_size)
            if microbatch_size != batch_size:
                mb_iter = tqdm(mb_iter, unit="ubatch", unit_scale=True, total=nr_micro_batches, leave=False)

            for i in mb_iter:
                microbatch = batch["img"][i : i + microbatch_size].to(device)
                t = torch.randint(0, schedule.timesteps, (microbatch.shape[0],), device=device).long()

                loss = p_losses(model, microbatch, t, schedule, loss_type="l2")
                loss_item = loss.item()
                loss_history.append(loss_item)

                loss.backward()

            if step % log_interval == 0:
                tqdm.write(f"Loss: {loss_item}")

            optimizer.step()
            ema_model.update_parameters(model)

            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                track_samples(out_path, epoch, milestone, model, microbatch_size, image_size, channels, sched=ddim_schedule)

            step += 1

        history[epoch] = loss_history
        save_all(epoch, model, ema_model, optimizer, unet_kwargs, history, schedule_kwargs, out_path)
        track_samples(out_path, epoch, "last", model, microbatch_size, image_size, channels, sched=ddim_schedule)

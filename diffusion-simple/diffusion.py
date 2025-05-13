import argparse
import math
from pathlib import Path
from collections import deque
import pickle
import os
import numpy as np

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T, models
import clip
from PIL import Image
from tqdm.auto import tqdm

from data import setup_loader
from utils.misc import exists, default, num_to_groups, latest_checkpoint
from utils.contextual_loss import ctx_loss_mod_forward_fused
from scheduling import Schedule, ScheduleDDIM, save_schedule_kwargs, load_schedule_kwargs, linear_beta_schedule, extract
from model import load_model, load_ema, load_optimizer, save_model, save_ema, save_optimizer, Unet


_clip_feature_maps = None
_clip_feature_maps_names = ("relu3", "layer1", "layer2", "layer3")

_vgg_feature_maps = None
_vgg_feature_maps_names = ("layer_8", "layer_17", "layer_26", "layer_35")
_vgg_feature_maps_weights = (1, 2, 3, 4)


def hook_clip(model):
    global _clip_feature_maps, _clip_feature_maps_names

    def get_activation(name):
        def hook(model, input, output):
            _clip_feature_maps[name] = output
        return hook

    hooks = []
    for name in _clip_feature_maps_names:
        if name == "relu3":
            layer = model.relu3
        else:
            layer = getattr(model, name)
        hooks.append(layer.register_forward_hook(get_activation(name)))

    return hooks


def extract_clip_features(model, input):
    global _clip_feature_maps, _clip_feature_maps_names
    _clip_feature_maps = {}
    with torch.no_grad():
        _ = model(input)
    clip_features_batch = _clip_feature_maps
    return clip_features_batch


def hook_vgg(model, device):
    global _vgg_feature_maps, _vgg_feature_maps_names

    def get_features(name):
        def hook(model, input, output):
            _vgg_feature_maps[name] = output
        return hook

    hooks = []
    for name in _vgg_feature_maps_names:
        layer_idx = int(name.split("_", 1)[1])
        layer = model.features[layer_idx]
        hooks.append(layer.register_forward_hook(get_features(name)))

    return hooks


def extract_vgg_features(model, input):
    global _vgg_feature_maps, _vgg_feature_maps_names
    _vgg_feature_maps = {}
    _ = model(input)
    vgg_features_batch = _vgg_feature_maps
    return vgg_features_batch


# forward diffusion (using the nice property)
def q_sample(x_start, t, sched, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sched.sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sched.sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, t, sched, grayscale=None, clip_features=None, ref=None, noise=None):
    global vgg_model, vgg_normalization, img_to_vgg, _vgg_feature_maps_names, _vgg_feature_maps_weights

    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, sched=sched, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, grayscale=grayscale, clip_features=clip_features)

    loss = F.mse_loss(noise, predicted_noise)
    ctx_loss = torch.zeros(t.size(0), device=t.device)

    if ref is not None:
        predicted_x_start = predict_start_from_noise(x_noisy, t, predicted_noise, sched)
        predicted_x_start_vgg = img_to_vgg(predicted_x_start)
        del predicted_x_start

        vgg_input = vgg_normalization(torch.cat([predicted_x_start_vgg, ref], dim=0))
        vgg_features = extract_vgg_features(vgg_model, vgg_input)

        batch_size = predicted_x_start_vgg.size(0)
        pred_features = {k: v[:batch_size] for k, v in vgg_features.items()}
        ref_features = {k: v[batch_size:] for k, v in vgg_features.items()}

        for layer_name, weight in zip(_vgg_feature_maps_names, _vgg_feature_maps_weights):
            ctx_loss_layer = ctx_loss_mod_forward_fused(
                x=pred_features[layer_name],
                y=ref_features[layer_name],
            )
            ctx_loss += (weight * ctx_loss_layer)

    return loss, ctx_loss.mean()


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
def p_sample(model, x, t, t_index, sched, grayscale=None, clip_features=None):
    # Equation 11 by using reconstructed x_0 in eq. 7.
    # Basically eq. 11 unsimplified, in order to clip intermediate x_0 into proper range,
    # using static thresholding.
    predicted_noise = model(x, t, grayscale=grayscale, clip_features=clip_features)
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
def p_sample_loop(model, shape, sched, grayscale=None, clip_features=None):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    imgs = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(0, sched.timesteps)), desc="Sampling loop time step", total=sched.timesteps, unit="step", leave=False):
        imgs = p_sample(model, imgs, torch.full((b,), i, device=device, dtype=torch.long), i, sched, grayscale=grayscale, clip_features=clip_features)
    return imgs


@torch.no_grad()
def ddim_sample(model, x, t, t_index, sched, eta=0.0, rederive_eps_from_recon=False, grayscale=None, clip_features=None):
    # DDPM Equation 11 by using reconstructed x_0 in eq. 7.
    # Basically eq. 11 unsimplified, in order to clip intermediate x_0 into proper range.
    eps = predicted_noise = model(x, sched.transform_times(t), grayscale=grayscale, clip_features=clip_features)
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
def ddim_sample_loop(model, shape, sched, grayscale=None, clip_features=None):
    device = next(model.parameters()).device

    b = shape[0]
    imgs = torch.randn(shape, device=device)

    for i in tqdm(reversed(range(0, sched.timesteps)), desc="Sampling loop time step", total=sched.timesteps, unit="step", leave=False):
        imgs = ddim_sample(model, imgs, torch.full((b,), i, device=device, dtype=torch.long), i, sched, grayscale=grayscale, clip_features=clip_features)
    return imgs


@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3, *, sched, grayscale=None, clip_features=None):
    if isinstance(sched, ScheduleDDIM):
        return ddim_sample_loop(model, shape=(batch_size, channels, image_size, image_size), sched=sched, grayscale=grayscale, clip_features=clip_features)
    else:
        return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), sched=sched, grayscale=grayscale, clip_features=clip_features)


def track_samples(
        folder, epoch, milestone, model,
        microbatch_size, image_size, channels,
        sched, grayscale=None, clip_features=None, ref=None,
    ):
    results_folder = folder / "results"
    results_folder.mkdir(exist_ok=True, parents=True)
    batches = num_to_groups(1, microbatch_size)

    if grayscale is None:
        sample_grayscale = torch.rand((1, 1, image_size, image_size), device=next(model.parameters()).device) * 2 - 1
        grayscale = sample_grayscale.repeat(microbatch_size, 1, 1, 1)

    model.eval()
    all_images_list = list(map(
        lambda n: sample(
            model,
            sched=sched,
            image_size=image_size,
            batch_size=n,
            channels=channels,
            grayscale=grayscale[:n] if grayscale is not None else None,
            clip_features={k: v[:n] for k, v in clip_features.items()} if clip_features is not None else None
        ),
        batches
    ))
    model.train()

    all_images = torch.cat(all_images_list, dim=0)
    all_images = (all_images + 1) / 2

    if grayscale is not None:
        grayscale_display = grayscale[:1].repeat(1, 3, 1, 1)
        grayscale_display = (grayscale_display + 1) / 2
        ref_display = T.functional.resize(ref[:1], size=(image_size, image_size))

        if ref is not None:
            comparison = torch.cat([grayscale_display, ref_display, all_images[:1]], dim=0)
            torchvision.utils.save_image(comparison, results_folder / f"comparison-{epoch}-{milestone}.png", nrow=1)
        else:
            comparison = torch.cat([grayscale_display, all_images[:1]], dim=0)
            torchvision.utils.save_image(comparison, results_folder / f"comparison-{epoch}-{milestone}.png", nrow=1)
    else:
        torchvision.utils.save_image(all_images, results_folder / f"sample-{epoch}-{milestone}.png", nrow=1)


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
    learning_rate = 3e-4
    weight_ctx_loss = 1

    epochs = 500
    batch_size = 32
    microbatch_size = 8

    timesteps = 1000
    ddim_timesteps = 25
    betas_f = linear_beta_schedule

    train_dset_path = Path(args.dataset) / "train_imgs" / f"resized_{image_size}"
    val_dset_path = Path(args.dataset) / "val_imgs" / f"resized_{image_size}"

    out_path = Path("./out") / name

    device = (
        torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.device("cpu")
    )
    unet_kwargs = {
        "image_size": image_size,
        "channels": channels,
        "init_dim": init_dim,
        "dim_mults": dim_mults,
    }

    # Load old or create new denoising U-Net
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

    # CLIP model for reference image features extraction
    resnet_full, _resnet_preprocess = clip.load("RN50", device="cpu")
    clip_model = resnet_full.visual
    clip_model.eval()
    clip_model.to(device)
    _clip_hooks = hook_clip(clip_model)

    # VGG model to provide feature maps for contextual loss between reference and output
    vgg_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    vgg_model.eval()
    vgg_model.to(device)
    _vgg_hooks = hook_vgg(vgg_model, device)

    reference_size = clip_model.input_resolution
    assert reference_size == 224  # VGG19 and CLIP ResNet50 training resolution

    # Input image to U-Net input
    img_transform = T.Compose([
        T.ToTensor(),  # (C x H x W) in the range [0.0; 1.0]
        T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),  # input is expected to have […, H, W] shape
        T.CenterCrop(image_size),  # input is expected to have […, H, W] shape
        T.Lambda(lambda x: x * 2 - 1),  # range [-1; 1]
    ])

    # U-Net output image tensor to VGG input
    img_to_vgg = T.Compose([
        T.Lambda(lambda x: (x + 1) / 2),  # range [-1; 1] -> [0; 1]
        T.Resize(reference_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(reference_size),
    ])

    # Reference image to CLIP and VGG input
    ref_transform = T.Compose([
        T.ToTensor(),
        T.Resize(reference_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(reference_size),
    ])

    clip_normalization = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    vgg_normalization = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_dataloader = setup_loader(
        data_dir=str(train_dset_path),
        batch_size=batch_size,
        shuffle=True,
        transforms={"img": img_transform, "ref": ref_transform},
    )
    nr_batches = math.ceil(len(train_dataloader.dataset) / batch_size)
    nr_micro_batches = math.ceil(batch_size / microbatch_size)
    save_and_sample_every = 10  # nth batch
    log_interval = 10

    schedule = Schedule(**schedule_kwargs)
    ddim_schedule = ScheduleDDIM(ddim_timesteps, schedule, **schedule_kwargs)

    save_all(0, model, ema_model, optimizer, unet_kwargs, history, schedule_kwargs, out_path)
    step = 1

    for epoch in tqdm(range(start_epoch, start_epoch + epochs), unit="epoch"):
        loss_history = deque()

        for batch in tqdm(train_dataloader, unit="batch", unit_scale=True, total=nr_batches, leave=False):
            optimizer.zero_grad()

            batch_size = batch["img"].shape[0]
            batch_nums = []
            batch_losses = []
            batch_ctx_losses = []

            mb_iter = range(0, batch_size, microbatch_size)
            if microbatch_size != batch_size:
                mb_iter = tqdm(mb_iter, unit="ubatch", unit_scale=True, total=nr_micro_batches, leave=False)

            for i in mb_iter:
                # Get color image, grayscale conditioning, reference conditioning.
                img = batch["img"][i : i + microbatch_size].to(device)
                grayscale = batch["grayscale"][i : i + microbatch_size].to(device)
                ref = batch["ref"][i : i + microbatch_size].to(device)

                this_mb_size = img.size(0)

                # Sample a time step for each image
                t = torch.randint(0, schedule.timesteps, (this_mb_size,), device=device).long()

                # Process each ref image in batch separately for CLIP
                clip_input = clip_normalization(ref)
                clip_features = extract_clip_features(clip_model, clip_input)

                loss, ctx_loss = p_losses(model, img, t, schedule, grayscale=grayscale, clip_features=clip_features, ref=ref)
                ctx_loss = weight_ctx_loss * ctx_loss
                unet_combined_loss = loss + ctx_loss
                unet_combined_loss.backward()

                loss_item = loss.item()
                ctx_loss_item = ctx_loss.item()
                loss_history.append((loss_item, ctx_loss_item))
                batch_nums.append(this_mb_size)
                batch_losses.append(loss_item)
                batch_ctx_losses.append(ctx_loss_item)

            optimizer.step()
            ema_model.update_parameters(model)

            if step % log_interval == 0:
                batch_loss = np.average(batch_losses, weights=batch_nums)
                batch_ctx_loss = np.average(batch_ctx_losses, weights=batch_nums)
                tqdm.write(f"{batch_loss=:.4f}, {batch_ctx_loss=:.4f}")

            if step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                track_samples(
                    out_path, epoch, milestone, model,
                    microbatch_size, image_size, channels,
                    sched=ddim_schedule, grayscale=grayscale, ref=ref, clip_features=clip_features,
                )
            step += 1

        history[epoch] = loss_history
        save_all(epoch, model, ema_model, optimizer, unet_kwargs, history, schedule_kwargs, out_path)
        track_samples(
            out_path, epoch, "last", model,
            microbatch_size, image_size, channels,
            sched=ddim_schedule, grayscale=grayscale, ref=ref, clip_features=clip_features,
        )

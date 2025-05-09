import argparse
import math
from pathlib import Path
from collections import deque
import pickle
import os
import clip
import matplotlib.pyplot as plt

from torchvision import transforms as T

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

def tensor_to_pil(img_tensor):
    # First convert from [-1,1] to [0,1]
    img_tensor = (img_tensor + 1) / 2
    # Clamp to ensure values are in valid range
    img_tensor = torch.clamp(img_tensor, 0, 1)
    # Convert to PIL
    return torchvision.transforms.ToPILImage()(img_tensor)

def visualize_feature_maps(feature_maps, num_features=4):
    plt.figure(figsize=(15, 10))

    for i, (layer_name, feature_map) in enumerate(feature_maps.items()):
        for j in range(min(num_features, feature_map.size(1))):
            plt.subplot(len(feature_maps), num_features, i * num_features + j + 1)
            plt.imshow(feature_map[0, j].cpu().detach().numpy(), cmap='viridis')
            plt.title(f"{layer_name} - ch{j}")
            plt.axis('off')

    plt.tight_layout()
    plt.savefig('RESNET_feature_maps.png')
    plt.show()

# extract resnet feature maps
def extract_clip_rn50_features(visual_model, input_image, layers_to_extract=None):

    if layers_to_extract is None:
        layers_to_extract = ['relu3', 'layer1', 'layer2', 'layer3']

    feature_maps = {}

    def get_activation(name):
        def hook(model, input, output):
            feature_maps[name] = output
        return hook

    hooks = []
    for name in layers_to_extract:
        if name == 'relu3':
            layer = visual_model.relu3
        else:
            layer = getattr(visual_model, name)
        hooks.append(layer.register_forward_hook(get_activation(name)))

    _ = visual_model(input_image)

    for hook in hooks:
        hook.remove()

    return feature_maps

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

def p_losses(denoise_model, x_start, t, sched, grayscale=None, clip_features=None, noise=None, loss_type="l2"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, sched=sched, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, grayscale=grayscale, clip_features=clip_features)

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
def p_sample(model, x, t, t_index, sched, grayscale=None, clip_features=None):
    # Equation 11 by using reconstructed x_0 in eq. 7.
    # Basically eq. 11 unsimplified, in order to clip intermediate x_0 into proper range.
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

def track_samples(folder, epoch, milestone, model, microbatch_size, image_size, channels, sched, grayscale=None, clip_features=None, exemplar=None):
    results_folder = folder / "results"
    results_folder.mkdir(exist_ok=True, parents=True)
    batches = num_to_groups(1, microbatch_size)

    if grayscale is None:
        sample_grayscale = torch.rand((1, 1, image_size, image_size), device=next(model.parameters()).device) * 2 - 1
        grayscale = sample_grayscale.repeat(microbatch_size, 1, 1, 1)

    all_images_list = list(map(
        lambda n: sample(
            model,
            sched=sched,
            image_size=image_size,
            batch_size=n,
            channels=channels,
            grayscale=grayscale[:n] if grayscale is not None else None,
            clip_features=clip_features[:n] if clip_features is not None else None
        ), 
        batches
    ))

    all_images = torch.cat(all_images_list, dim=0)
    all_images = (all_images + 1) / 2

    if grayscale is not None:
        grayscale_display = grayscale[:1].repeat(1, 3, 1, 1)
        grayscale_display = (grayscale_display + 1) / 2
        
        if exemplar is not None:
            exemplar_display = (exemplar[:1] + 1) / 2
            comparison = torch.cat([grayscale_display, exemplar_display, all_images[:1]], dim=0)
            torchvision.utils.save_image(comparison, results_folder / f"comparison-{epoch}-{milestone}.png", nrow=1)
        else:
            comparison = torch.cat([grayscale_display, all_images[:1]], dim=0)
            torchvision.utils.save_image(comparison, results_folder / f"comparison-{epoch}-{milestone}.png", nrow=1)

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
    learning_rate = 1e-4
    epochs = 500
    batch_size = 256
    microbatch_size = 64
    save_and_sample_every = 250  # nth batch
    log_interval = 50

    timesteps = 500
    ddim_timesteps = 25
    betas_f = linear_beta_schedule

    train_dset_path = Path(args.dataset) / "train_imgs" / f"resized_{image_size}"
    val_dset_path = Path(args.dataset) / "val_imgs" / f"resized_{image_size}"
    # train_dset_path = Path(args.dataset) / "cifar_train"
    # val_dset_path = Path(args.dataset) / "cifar_test"

    # train_dset_path = Path(args.dataset) / "train_imgs"
    # val_dset_path = Path(args.dataset) / "val_imgs"

    out_path = Path("./out") / name

    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Lambda(lambda x: x * 2 - 1),
    ])

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

    resnet_full, resnet_preprocess = clip.load("RN50", device="cpu")
    resnet_visual = resnet_full.visual
    resnet_visual.train()
    resnet_visual.to(device)

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
                # Get both the color image and grayscale conditioning
                microbatch = batch["img"][i : i + microbatch_size].to(device)
                grayscale = batch["grayscale"][i : i + microbatch_size].to(device)

                # exemplar = batch["exemplar"][i : i + microbatch_size].to(device)
                # Get exemplar image (could be the same as input image or different)
                exemplar = batch["ref"][i : i + microbatch_size].to(device)
                
                # Process each exemplar image separately for CLIP
                feature_maps_batch = {}
                selected_layers = ['relu3', 'layer1', 'layer2', 'layer3']
                
                # Process the batch
                all_features_list = []
                for j in range(exemplar.shape[0]):
                    pil_image = tensor_to_pil(exemplar[j])
                    clip_input = resnet_preprocess(pil_image).unsqueeze(0).to(device)
                    
                    features = extract_clip_rn50_features(resnet_visual, clip_input, selected_layers)
                    
                    all_features_list.append(features)

                clip_features = {}
                for layer_name in selected_layers:
                    layer_features = torch.cat([feat[layer_name] for feat in all_features_list], dim=0)
                    clip_features[layer_name] = layer_features
                
                t = torch.randint(0, schedule.timesteps, (microbatch.shape[0],), device=device).long()

                # Pass grayscale to p_losses
                loss = p_losses(model, microbatch, t, schedule, grayscale=grayscale, loss_type="l2")
                loss_item = loss.item()
                loss_history.append(loss_item)

                loss.backward()

            optimizer.step()
            ema_model.update_parameters(model)

            if step != 0 and step % save_and_sample_every == 0:
                milestone = step // save_and_sample_every
                track_samples(out_path, epoch, milestone, model, microbatch_size, image_size, channels, sched=ddim_schedule, grayscale=grayscale, exemplar=exemplar)
            step += 1

        history[epoch] = loss_history
        save_all(epoch, model, ema_model, optimizer, unet_kwargs, history, schedule_kwargs, out_path)
        track_samples(out_path, epoch, "last", model, microbatch_size, image_size, channels, sched=ddim_schedule, grayscale=grayscale, exemplar=exemplar)

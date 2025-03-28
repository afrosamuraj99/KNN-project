from safe_gpu import safe_gpu

safe_gpu.claim_gpus()

import random
from pathlib import Path

import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything

from utils.data import HWC3, apply_color
from utils.ddim import DDIMSampler
from utils.model import create_model, load_state_dict

model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict('./checkpoints/colorizenet-sd21.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

out_imgs_dir = Path("./out_imgs/512")
val_imgs_dir = Path("./val_imgs/512")
H, W, C = (512, 512, 3)
batch_size = 8
ddim_steps = 25
num_samples = 1
seed = 1623808298

prompt = "Colorize this image"
n_prompt = ""
guess_mode = False
strength = 1.0
eta = 0.0
scale = 9.0

out_imgs_dir.mkdir(parents=False, exist_ok=True)
seed_everything(seed)
model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)

batch_num = 0
total_num = 0
imgs_iter = val_imgs_dir.iterdir()
stop = False
while not stop:
    names = []
    images = []
    batch = []
    batch_num += 1
    for i in range(batch_size):
        try:
            img_path = next(imgs_iter)
            names.append(img_path.stem)
        except StopIteration:
            stop = True
            break
        img = cv2.imread(str(img_path))
        img = HWC3(img)
        assert img.shape == (H, W, C)
        images.append(img)
        control = torch.from_numpy(img.copy()).float() / 255.0
        batch.extend([control for _ in range(num_samples)])

    num_imgs = len(batch)
    if stop and num_imgs == 0:
        break

    total_num += num_imgs
    real_batch_size = num_samples * num_imgs

    control = torch.stack(batch, dim=0).cuda()
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * real_batch_size)]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * real_batch_size)]}
    shape = (4, H // 8, W // 8)

    samples, intermediates = ddim_sampler.sample(ddim_steps, real_batch_size,
                                                shape, cond, verbose=False, eta=eta,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=un_cond)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    for img_idx, (name, img) in enumerate(zip(names, images)):
        i = img_idx * num_samples
        results = [x_samples[i + j] for j in range(num_samples)]
        colored_results = [apply_color(img, result) for result in results]
        [cv2.imwrite(str(out_imgs_dir / f"{name}_{i}.jpg"), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            for i, result in enumerate(colored_results)]

    print(f"batch {batch_num} | total images {total_num}")

print("Finished")

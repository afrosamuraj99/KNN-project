import math
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from utils.misc import default, exists


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )

def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    # return nn.Sequential(
    #     Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
    #     nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    # )
    return nn.Sequential(
            nn.Conv2d(dim, default(dim_out, dim), 1, stride=2),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
        self,
        image_size,
        channels,
        init_dim,
        dim_mults,
        resnet_block_groups=4,
        grayscale_channels=1,
        reference_channels=3,
    ):
        super().__init__()

        self.image_size = image_size
        self.channels = channels
        self.grayscale_channels = grayscale_channels

        input_channels = channels + grayscale_channels
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = init_dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(init_dim),
            nn.Linear(init_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        self.clip_names = ("relu3", "layer1", "layer2", "layer3")
        self.clip_channels = [64, 256, 512, 1024]
        self.clip_sizes = [112, 56, 28, 14]
        self.clip_paddings = []

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.down_feature_convs = nn.ModuleList([])
        self.up_feature_convs = nn.ModuleList([])

        num_resolutions = len(in_out)
        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= (num_resolutions - 1)

            # print(f"\nCLIP layer {idx} feature size: {clip_size}")
            # Calculate padding - use x.shape for first block, h[-2] for others
            h_x = self.image_size // (2 ** idx)
            w_x = h_x
            h_feat = self.clip_sizes[idx]
            w_feat = h_feat
            # print(f"Target shape for feature {i}: ({h_x}, {w_x})")
            assert (h_feat < h_x) and (w_feat < w_x)
            pad_h = (h_x - h_feat)
            pad_w = (w_x - w_feat)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            padding = (pad_left, pad_right, pad_top, pad_bottom)
            self.clip_paddings.append(padding)
            # print(f"Padding for feature {i}: (left={pad_left}, right={pad_right}, top={pad_top}, bottom={pad_bottom})")

            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in + reference_channels, dim_in, time_emb_dim=time_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                    # Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out)
                    if not is_last
                    else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ])
            )

            # down feature mapa
            self.down_feature_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.clip_channels[idx], reference_channels, kernel_size=1),
                    nn.ZeroPad2d(padding)
                )
            )

            # up feature mapa
            self.up_feature_convs.append(
                nn.Sequential(
                    nn.Conv2d(self.clip_channels[idx], reference_channels, kernel_size=1),
                    nn.ZeroPad2d(padding)
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in + reference_channels, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        # Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = block_klass(init_dim * 2, init_dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(init_dim, channels, 1)

    def forward(self, x, time, grayscale=None, clip_features=None):
        # print(f"CLIP features structure:")
        # for k, v in clip_features.items():
            # print(f"{k}: {v.shape}")

        # Debug initial inputs
        # print(f"\n=== Initial Inputs ===")
        # print(f"Input x shape: {x.shape}")
        # print(f"Time shape: {time.shape}")
        # if grayscale is not None:
            # print(f"Grayscale shape: {grayscale.shape}")
        # if clip_features is not None:
            # print(f"CLIP features keys: {clip_features.keys()}")

        if grayscale is not None:
            x = torch.cat((x, grayscale), dim=1)
            # print(f"After grayscale concatenation, x shape: {x.shape}")

        # Initial convolution
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        h = []
        # print(f"\nAfter init_conv - x shape: {x.shape}")
        # print(f"Time embedding shape: {t.shape}")

        # Process CLIP features
        clip_maps = []
        if clip_features is not None:
            # print("\n=== Processing CLIP Features ===")
            for i, (conv_down, conv_up) in enumerate(zip(self.down_feature_convs, self.up_feature_convs)):
                layer_name = self.clip_names[i]
                feat = clip_features[layer_name]
                # print(f"\nCLIP layer {layer_name} feature shape: {feat.shape}")
                # Process features
                down_feat = conv_down(feat)
                up_feat = conv_up(feat)
                # print(f"Processed down feature {i} shape: {down_feat.shape}")
                # print(f"Processed up feature {i} shape: {up_feat.shape}")
                clip_maps.append((down_feat, up_feat))

        # Downsample path
        # print("\n=== Downsample Path ===")
        for i, (block1, block2, downsample) in enumerate(self.downs):
            # print(f"\nDown block {i}")
            # print(f"Input shape: {x.shape}")

            # Add down feature if available
            if clip_features is not None and i < len(clip_maps):
                feat, _ = clip_maps[i]
                # print(f"Adding down feature {i} with shape: {feat.shape}")
                x = torch.cat([x, feat], dim=1)
                # print(f"After feature concatenation: {x.shape}")

            x = block1(x, t)
            # print(f"After block1: {x.shape}")
            h.append(x)

            x = block2(x, t)
            # print(f"After block2: {x.shape}")
            h.append(x)

            x = downsample(x)
            # print(f"After downsample: {x.shape}")

        # Middle blocks
        # print("\n=== Middle Blocks ===")
        # print(f"Input shape: {x.shape}")
        x = self.mid_block1(x, t)
        # print(f"After mid_block1: {x.shape}")
        x = self.mid_attn(x)
        # print(f"After attention: {x.shape}")
        x = self.mid_block2(x, t)
        # print(f"After mid_block2: {x.shape}")

        # Upsample path
        # print("\n=== Upsample Path ===")
        for i, (block1, block2, upsample) in enumerate(self.ups):
            # print(f"\nUp block {i}")
            # print(f"Current x shape: {x.shape}")
            # print(f"Popping from h (shape: {h[-1].shape})")

            x = torch.cat((x, h.pop()), dim=1)
            # print(f"After first cat with h: {x.shape}")

            # Add up feature if available
            reverse_idx = len(self.ups)-1-i
            if clip_features is not None and reverse_idx < len(clip_maps):
                _, feat = clip_maps[reverse_idx]
                # print(f"Adding up feature {reverse_idx} with shape: {feat.shape}")
                x = torch.cat([x, feat], dim=1)
                # print(f"After feature concatenation: {x.shape}")

            x = block1(x, t)
            # print(f"After block1: {x.shape}")

            # print(f"Popping from h (shape: {h[-1].shape})")
            x = torch.cat((x, h.pop()), dim=1)
            # print(f"After second cat with h: {x.shape}")

            x = block2(x, t)
            # print(f"After block2: {x.shape}")

            x = upsample(x)
            # print(f"After upsample: {x.shape}")

        # Final processing
        # print("\n=== Final Processing ===")
        # print(f"x shape: {x.shape}")
        # print(f"r shape: {r.shape}")
        x = torch.cat((x, r), dim=1)
        # print(f"After final cat with r: {x.shape}")
        x = self.final_res_block(x, t)
        # print(f"After final res block: {x.shape}")
        output = self.final_conv(x)
        # print(f"Final output shape: {output.shape}")
        return output


def save_model(model, unet_kwargs, path):
    for attr in ["grayscale_channels", "reference_channels"]:
        if hasattr(model, "attr") and "attr" not in unet_kwargs:
            unet_kwargs["attr"] = model.grayscale_channels

    torch.save({
        "model_state_dict": model.state_dict(),
        "unet_kwargs": unet_kwargs,
    }, path)


def save_ema(ema_model, ema_decay, unet_kwargs, path):
    torch.save({
        "model_state_dict": ema_model.state_dict(),
        "ema_decay": ema_decay,
        "unet_kwargs": unet_kwargs,
    }, path)


def save_optimizer(optimizer, path):
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_model(path, model_class, mode):
    checkpoint = torch.load(path, weights_only=True, mmap=False)
    with torch.device("meta"):
        model = model_class(**checkpoint["unet_kwargs"])
    model.load_state_dict(checkpoint["model_state_dict"], assign=True)
    return model


def load_ema(path, model_class, mode):
    checkpoint = torch.load(path, weights_only=True, mmap=False)

    with torch.device("meta"):
        model = model_class(**checkpoint["unet_kwargs"])
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


def load_optimizer(path, model):
    checkpoint = torch.load(path, weights_only=True, mmap=False)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return optimizer

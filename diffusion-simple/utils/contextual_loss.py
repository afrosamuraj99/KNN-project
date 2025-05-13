"""
Implementation of:
    The Contextual Loss for Image Transformation with Non-Aligned Data
    (https://arxiv.org/abs/1803.02077)

Modified from:
    https://github.com/roimehrez/contextualLoss
"""

import torch
from einops import rearrange, reduce


def center_by_y(y, x):
    assert y.shape == x.shape
    mean_y = reduce(y, "n h w c -> n 1 1 c", "mean")
    y_centered = y - mean_y
    x_centered = x - mean_y
    return y_centered, x_centered


def l2_normalize_channelwise(features):
    def reduction(tensor, axs):
        return torch.linalg.vector_norm(tensor, ord=2, dim=axs, keepdim=False)

    norms = reduce(features, "n h w c -> n h w 1", reduction)
    features = features / norms
    return features


def patch_decomposition(y):
    # 1HWC --> 11PC --> PC11, with P=H*W
    patches_PC11 = rearrange(y, "1 h w c -> (h w) c 1 1")
    return patches_PC11


def relative_distances(raw_distances, axis=3):
    epsilon = 1e-5
    div, _idxs = torch.min(raw_distances, dim=axis, keepdim=True)
    relative_dist = torch.div(raw_distances, div + epsilon)
    return relative_dist


def normalized_similarities(scaled_distances, b=1.0, h=0.1, axis_for_normalization=3):
    cs_NHWC = torch.nn.functional.softmax(
        (b - scaled_distances) / h,
        dim=axis_for_normalization,
    )
    return cs_NHWC


def ctx_loss_mod_forward_fused(x, y):
    x = rearrange(x, "n c h w -> n h w c")
    y = rearrange(y, "n c h w -> n h w c")
    y, x = center_by_y(y, x)
    x = l2_normalize_channelwise(x)
    y = l2_normalize_channelwise(y)
    # Work seperatly for each pair (x_i, y_i) of examples in dim 1, i == j
    cx_loss_all = []
    count = y.size(0)
    for i in range(count):
        # One y image and one x image
        y_i = y[i, :, :, :].unsqueeze(0)
        # Convolution input formatting, 1HWC --> 1CHW
        x_i = x[i, :, :, :].unsqueeze(0).permute((0, 3, 1, 2))
        # Convolution filter formatting, 1HWC --> PC11, with P=H*W
        # Pixels (out dim) -> Channels = (in dim) -> 1 = (kx) -> 1 = (ky)
        patches_PC11_i = patch_decomposition(y_i)
        cosine_sim_i_1CHW = torch.nn.functional.conv2d(x_i, patches_PC11_i)
        del patches_PC11_i
        cosine_sim_i_1HWC = cosine_sim_i_1CHW.permute((0, 2, 3, 1))
        del cosine_sim_i_1CHW
        raw_dist = 1 - cosine_sim_i_1HWC
        del cosine_sim_i_1HWC
        relative_dist = relative_distances(raw_dist)
        del raw_dist
        cs = normalized_similarities(relative_dist)
        del relative_dist
        # ((H W) is the x dim, C is the y dim)
        k_max_NC = reduce(cs, "n h w c -> n (h w)", "max")
        cs = torch.mean(k_max_NC, dim=1)
        del k_max_NC
        cx_loss = -torch.log(cs)
        del cs
        cx_loss_all.append(cx_loss)
        del cx_loss
    cx_loss = torch.cat(cx_loss_all, dim=0)
    del cx_loss_all
    return cx_loss


def test():
    import numpy as np
    torch.autograd.set_detect_anomaly(True)

    a = torch.tensor(1., requires_grad=True)
    b = torch.tensor(1., requires_grad=True)

    x = a * torch.tensor([
        # [
        #     [[1,2,3], [4,5,6], [7,8,9]], # row 1
        #     [[9,10,11], [12,13,14], [15,16,17]], # row 2
        #     [[18,19,20], [21,22,23], [24,25,26]], # row 3
        # ],
        [
            [[1,2,3], [1,2,3], [1,2,3]], # row 1
            [[9,9,9], [9,9,9], [9,9,9]], # row 2
            [[1,2,3], [1,2,3], [1,2,3]], # row 3
        ],
        [
            [[1,2,3], [1,2,3], [1,2,3]], # row 1
            [[8,8,8], [8,8,8], [8,8,8]], # row 2
            [[1,2,3], [1,2,3], [1,2,3]], # row 3
        ],
        # [
        #     [[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]], # row 1
        #     [[9,9,9,9,9], [9,9,9,9,9], [9,9,9,9,9]], # row 2
        #     [[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]], # row 3
        # ],
        # [
        #     [[2,3,4,5,6], [2,3,4,5,6], [2,3,4,5,6]], # row 1
        #     [[2,3,4,5,6], [2,3,4,5,6], [2,3,4,5,6]], # row 3
        #     [[2,3,4,5,6], [2,3,4,5,6], [2,3,4,5,6]], # row 2
        # ],
    ], dtype=torch.float32)
    # noise = torch.tensor(np.random.normal(loc=0, scale=25, size=x.shape), dtype=torch.float32)
    # x = x + noise
    y = b * (x.detach().clone())  # Should lead to CX = 1 for each (x_i, y_i) pair
    # y = torch.cat([x[None, 1], x[None, 0]]) # Should lead to CX != 1 for each (x_i, y_i) pair
    # y = x[None, 1]
    # x = x[None, 0]
    x_NCHW = rearrange(x, "n h w c -> n c h w")
    y_NCHW = rearrange(y, "n h w c -> n c h w")
    ctx_loss = ctx_loss_mod_forward_fused(x_NCHW, y_NCHW)
    ctx_loss.mean().backward()


if __name__ == "__main__":
    test()

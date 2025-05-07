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


def create_using_dotP(x, y, sigma=0.1, b=1.0):
    # Work seperatly for each pair (x_i, y_i) of examples in dim 1, i == j
    cosine_sim_l = []
    count = y.size(0)
    for i in range(count):
        # One y image and one x image
        y_i = y[i, :, :, :].unsqueeze_(0)
        # Convolution input formatting, 1HWC --> 1CHW
        x_i = x[i, :, :, :].unsqueeze_(0).permute((0, 3, 1, 2))
        # Convolution filter formatting, 1HWC --> PC11, with P=H*W
        # Pixels (out dim) -> Channels = (in dim) -> 1 = (kx) -> 1 = (ky)
        patches_PC11_i = patch_decomposition(y_i)
        cosine_sim_i_1CHW = torch.nn.functional.conv2d(x_i, patches_PC11_i)
        cosine_sim_i_1HWC = cosine_sim_i_1CHW.permute((0, 2, 3, 1))
        cosine_sim_l.append(cosine_sim_i_1HWC)

    cosine_sim = torch.cat(cosine_sim_l, dim=0)
    raw_dist = 1 - cosine_sim
    relative_dist = relative_distances(raw_dist)
    cs = normalized_similarities(relative_dist)

    return cs, relative_dist, raw_dist


def get_feature_similarities(x, y):
    x = rearrange(x, "n c h w -> n h w c")
    y = rearrange(y, "n c h w -> n h w c")
    y_centered, x_centered = center_by_y(y, x)
    x_centered_normalized = l2_normalize_channelwise(x_centered)
    y_centered_normalized = l2_normalize_channelwise(y_centered)
    cs, relative_dist, raw_dist = create_using_dotP(x_centered_normalized, y_centered_normalized)
    return cs


def similarities_to_loss_og_backward(similarities):
    """
    The Contextual Loss for Image Transformation with Non-Aligned Data (https://arxiv.org/abs/1803.02077)
    > To calculate the similarity between the images,
    > we find for each feature y_j the feature x_i that
    > is most similar to it, and then sum the corresponding
    > feature similarity values over all y_j.
    """
    # ((H W) is the x dim, C is the y dim)
    k_max_NC = reduce(similarities, "n h w c -> n c", "max")
    cs = torch.mean(k_max_NC, dim=1)
    cx_loss = -torch.log(cs)
    return cx_loss


def similarities_to_loss_mod_forward(similarities):
    """
    Deep Exemplar-based Video Colorization (https://arxiv.org/abs/1906.09909)
    > Contrary to the backward matching in [45], we
    > use forward matching where for each feature x_i
    > we find the closest feature y_j.
    > This is because some objects in x may not exist in y.
    """
    # ((H W) is the x dim, C is the y dim)
    k_max_NC = reduce(cs, "n h w c -> n (h w)", "max")
    cs = torch.mean(k_max_NC, dim=1)
    cx_loss = -torch.log(cs)
    return cx_loss


def test():
    import numpy as np

    x = torch.tensor([
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
    noise = torch.tensor(np.random.normal(loc=0, scale=25, size=x.shape), dtype=torch.float32)
    x = x + noise
    y = x.detach().clone()  # Should lead to CX = 1 for each (x_i, y_i) pair
    # y = torch.cat([x[None, 1], x[None, 0]]) # Should lead to CX != 1 for each (x_i, y_i) pair
    # y = x[None, 1]
    # x = x[None, 0]
    print("\nx")
    print(x)

    y_centered, x_centered = center_by_y(y, x)
    print("\nx_centered")
    print(x_centered)

    x_centered_normalized = l2_normalize_channelwise(x_centered)
    y_centered_normalized = l2_normalize_channelwise(y_centered)
    print("\nx_centered_normalized")
    print(x_centered_normalized)

    cs, relative_dist, raw_dist = create_using_dotP(x_centered_normalized, y_centered_normalized)
    print("Real CD")
    print(raw_dist)
    print("Relative")
    print(relative_dist)
    print("Softmaxed")
    print(cs)

    print("\nFinalization original")
    k_max_NC = reduce(cs, "n h w c -> n c", "max")
    print(k_max_NC)
    CS_og = torch.mean(k_max_NC, dim=1)
    CX_loss_og = -torch.log(CS_og)
    print(CS_og)

    print("\nFinalization forward")
    k_max_NC = reduce(cs, "n h w c -> n (h w)", "max")
    print(k_max_NC)
    CS_fw = torch.mean(k_max_NC, dim=1)
    CX_loss_fw = -torch.log(CS_fw)
    print(CS_fw)


if __name__ == "__main__":
    test()

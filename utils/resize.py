from pathlib import Path
import argparse

from PIL import Image
import numpy as np


def center_crop_arr(pil_image, image_size):
    """
    https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/image_datasets.py#L126
    """
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="path to the dataset directory")
    args = ap.parse_args()

    dset_path = Path(args.dataset)
    original_path = dset_path / "original"
    resized128_path = dset_path / "resized_128"
    resized512_path = dset_path / "resized_512"

    resized128_path.mkdir(parents=False, exist_ok=False)
    resized512_path.mkdir(parents=False, exist_ok=False)

    for og_img in original_path.iterdir():
        print("Processing", og_img)
        pil_image = Image.open(og_img)
        pil_image.load()
        pil_image = pil_image.convert("RGB")
        r128 = center_crop_arr(pil_image, 128)
        r512 = center_crop_arr(pil_image, 512)
        Image.fromarray(r128.astype('uint8'), 'RGB').save(resized128_path / og_img.name)
        Image.fromarray(r512.astype('uint8'), 'RGB').save(resized512_path / og_img.name)


if __name__ == "__main__":
    main()

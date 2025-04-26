import argparse
from pathlib import Path

import numpy as np
from PIL import Image

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    args = ap.parse_args()

    out_path = Path("./logs-sample/images")
    out_path.mkdir(exist_ok=True, parents=True)
    for f_path in out_path.iterdir():
        f_path.unlink()

    npz = np.load(args.npz)
    imgs_buf = npz["arr_0"]

    for i in range(imgs_buf.shape[0]):
        img = Image.fromarray(imgs_buf[i], "RGB")
        img.save(str(out_path / f"{i}.png"))

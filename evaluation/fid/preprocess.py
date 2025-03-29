"""
Prepare raw image set for evaluation, creating an .npz file for the evaluate.py script.

Example usage:
    Prepare reference input:
        python preprocess.py ../datasets/coco/train_imgs/resized_512 data/coco_train_512_10k.npz 512 10000

    Prepare generated input:
        python preprocess.py ../models/ColorizeNet/out_imgs/512 data/colorizenet_coco_val_512_all.npz 512 all

    Evaluation:
        python evaluate.py data/coco_train_512_10k.npz data/colorizenet_coco_val_512_all.npz | tee results/coco_train_512_10k_colorizenet_coco_val_512_all.txt
"""

import argparse
import random
import secrets
from pathlib import Path
from collections import deque

import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image

from evaluator import Evaluator, BatchIterator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref_dir", help="path to reference images dir")
    parser.add_argument("ref_batch", help="path to output reference batch npz file")
    parser.add_argument("img_size", help="resolution of one side of the square images", type=int)
    parser.add_argument("num_samples", help="number of samples to store in npz file, or 'all'")
    parser.add_argument("--batch-size", help="activations batch size", type=int, default=256)
    parser.add_argument("--random", help="sample in stochastic fashion, if num_samples is not 'all'", action="store_true")
    args = parser.parse_args()

    config = tf.ConfigProto(
        allow_soft_placement=True
    )
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))

    print("warming up TensorFlow...")
    evaluator.warmup()

    if args.random:
        random.seed(secrets.randbits(k=32))
    else:
        random.seed(1298435877)

    samples = deque()
    img_size = args.img_size
    nr_samples = args.num_samples

    try:
        if nr_samples != "all":
            nr_samples = int(nr_samples, base=10)
    except ValueError:
        print("Number of samples argument must be a number or 'all', not", nr_samples)
        exit(1)

    batcher = Batcher(args.ref_dir, samples, nr_samples, args.batch_size, img_size)
    if nr_samples == "all":
        nr_samples = batcher.ref_count

    print("computing reference batch activations...")
    ref_acts = evaluator.compute_activations(batcher.batches())

    print("computing reference batch statistics...")
    ref_stats, ref_stats_spatial = tuple(evaluator.compute_statistics(x) for x in ref_acts)

    samples = np.array(samples)
    print("Samples shape:", samples.shape)
    assert samples.shape == (nr_samples, img_size, img_size, 3)
    np.savez(args.ref_batch, samples,
             mu=ref_stats.mu, sigma=ref_stats.sigma,
             mu_s=ref_stats_spatial.mu, sigma_s=ref_stats_spatial.sigma)


class Batcher():
    def __init__(self, ref_dir, samples, nr_samples, batch_size, img_size):
        ref_imgs_paths = list(Path(ref_dir).iterdir())
        self.ref_idx = 0
        self.ref_count = len(ref_imgs_paths)
        self.source = list(enumerate(ref_imgs_paths))
        if nr_samples == "all":
            self.sample_idxs = deque(range(self.ref_count))
        else:
            self.sample_idxs = deque(sorted(random.sample(range(self.ref_count), k=nr_samples)))
        self.samples = samples
        self.batch_size = batch_size
        self.img_size = img_size

    def batches(self):
        def gen_fn():
            while True:
                batch = self.read_batch()
                if batch is None:
                    break
                yield batch

        rem = self.remaining()
        num_batches = rem // self.batch_size + int(rem % self.batch_size != 0)
        return BatchIterator(gen_fn, num_batches)


    def read_batch(self):
        if self.ref_idx >= self.ref_count:
            return None

        batch = deque()
        bs = min(self.batch_size, self.ref_count - self.ref_idx)

        batch_source = self.source[self.ref_idx : self.ref_idx + bs]
        for idx, img_path in batch_source:
            im = Image.open(img_path)
            img = np.array(im)
            batch.append(img)

            if self.sample_idxs and idx == self.sample_idxs[0]:
                self.sample_idxs.popleft()
                self.samples.append(img)

            assert im.mode == "RGB"
            assert im.size == (self.img_size, self.img_size)
            assert img.shape == (self.img_size, self.img_size, 3)

        self.ref_idx += bs
        return np.array(batch)


    def remaining(self):
        return max(0, self.ref_count - self.ref_idx)


if __name__ == "__main__":
    from safe_gpu import safe_gpu
    safe_gpu.claim_gpus(1, safe_gpu.TensorflowPlaceholder)
    main()

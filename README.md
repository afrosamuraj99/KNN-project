# KNN-project

Everything was tested on university student servers for GPU computing.

## Dependencies

Python version used was 3.10.
Dependencies are managed through the uv package manager.
To setup your environment, install uv as described here [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer).
Then run `uv sync`, which will download all training dependencies.

All training and sampling scripts are in the `diffusion-simple` folder.
To run any script, be it training or sampling or other, always run with the following command template:

```sh
CUDA_VISIBLE_DEVICES="x" uv run python script.py --args
```

## Training

To train a model run the diffusion script like so:

```sh
CUDA_VISIBLE_DEVICES="x" uv run python diffusion.py --name experiment1 --resolution 128 --dataset ../../datasets/coco --big --lr 1e-4
```

where `../datasets/coco` is a path you choose yourself, it needs to have the following structure:

```
- train_imgs
    - resized_128
    - resized_128_references
- val_imgs
    - resized_128
    - resized_128_references
```

where the `resized_128` folders contain images resized to the given resolution from the given dataset split,
and `resized_128_references` contain image files with the same names except it's their nearest neighbour as described in the report.
So `resized_128/cat1.jpg` is some image and `resized_128_references/cat1.jpg` is the image considered it's nearest neighbour.

For resizing look at the `--help` of the script `scripts/resize.py`.
For nearest neighbours look at the `--help` of the script `scripts/find_nearest.py`.

## Sampling

```sh
CUDA_VISIBLE_DEVICES="x" uv run python sample.py --name experiment1 --resolution 128 --ddpm --colored ../../datasets/coco/val_imgs/resized_128 --batch_size 8 --num_samples 8 --out samples.jpg --reference --big --seed 1337
```

For other options, look at `--help`.


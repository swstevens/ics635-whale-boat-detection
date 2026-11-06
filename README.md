# Whale and Boat Detection in Planet SkySat Imagery

ICS 635 semester project. Team: Shea Stevens, Joseph Behrens and Kim Cates.
Collaborator: Jake Linksy, PhD (HIMB, Madin lab).

This repo gives each 500x500 SkySat chip one label: **water**, **whale** or **boat**. It trains and
compares three models: EfficientNet-B0, DeiT-Tiny, and a SatViT-V2 encoder with a classification
head.

## Setup

```bash
uv sync
uv run pre-commit install   # once per clone
uv run jupyter lab
```

Without uv, follow the steps at the top of `requirements.txt`.

## Running it

1. **Get the data.** Run `notebooks/data_ingester.ipynb` in Colab. It downloads the chips and masks
   from Google Drive. `train_chip_classifier.ipynb` reads them from `data/whale/data/`:

   ```
   data/whale/data/images/000000000035.tif
   data/whale/data/labels/boats/000000000035.tif
   data/whale/data/labels/whales/000000000447.tif
   ```

2. **Get the SatViT weights (optional).** Download `SatViT_V2.pt` into `checkpoints/`, using the
   links in `satvit/README.md`. Without the file, the SatViT run is skipped and the other two models
   still train.
3. **Train.** Run `notebooks/train_chip_classifier.ipynb`. It saves each model's best weights to
   `checkpoints/` and ends with a comparison table.

## Code style

[Ruff](https://docs.astral.sh/ruff/) enforces PEP 8 style (100-character
lines), catches unused imports and undefined names, and sorts imports.

- The **pre-commit hook** runs ruff and strips notebook outputs on every commit.
  If it changes a file, the commit stops; `git add` the fixes and commit again.
- **GitHub Actions** runs the same ruff checks on every push and pull request.
- To run it by hand: `uv run ruff check --fix . && uv run ruff format .`

## Layout

```
├── notebooks/
│   ├── data_ingester.ipynb          downloads chips and masks from Google Drive
│   └── train_chip_classifier.ipynb  trains and compares the three models
├── satvit/            SatViT-V2 encoder code (see satvit/README.md for the weights)
├── checkpoints/       model weights (git-ignored)
└── data/              imagery (git-ignored)
```

# Whale and Boat Detection in Planet SkySat Imagery

ICS 635 semester project. Team: Shea Stevens, Joseph Behrens and Kim Cates.
Collaborator: Jake Linksy, PhD (HIMB, Madin lab).

This repo gives each 500x500 SkySat chip one label: **water**, **whale** or **boat**. It trains and
compares three models (EfficientNet-B0, DeiT-Tiny, and a SatViT-V2 encoder with a classification
head), then evaluates them against simple baselines.

## Setup

```bash
uv sync
uv run pre-commit install   # once per clone
uv run jupyter lab
```

Without uv, follow the steps at the top of `requirements.txt`. The scripts that build the SkySat
chips need two extra packages: `uv sync --group data` (or `pip install pyshp pandas`).

## Data

Two chip sets, both 500x500, 4 bands (Blue, Green, Red, NIR), uint16, 0.5 m, georeferenced:

| Set | Where | What |
| --- | --- | --- |
| original | `data/whale/data/` | 144 hand-picked chips with whale/boat masks |
| skysat | `data/skysat_chips_500/` | ~2,000 chips cut from the annotated SkySat scenes |

`data_ingester.ipynb` downloads both from Google Drive. `skysat_chips_500/` holds:

```
main/               training pool: water, boat, whales with certainty 1-2
uncertain_whales/   "possibly a whale" (certainty 0); evaluation only, never trained on
set_aside/          duplicates and water chips with a possible missed object (with reasons)
qc/                 quality-check tables
```

Each folder has `images/`, `manifest.csv` (one row per chip: label, scene, date, quality
stats), `points.csv` (each annotated object's pixel position and annotator certainty) and
`manifest_conditions.csv` (sun angle, haze, water roughness).

**Where the SkySat chips come from.** The source scenes and the annotators' point shapefiles
(`SatWhale/WhaleDetect/`) are on the lab's external drive. The labels are one annotator's visual
judgment per scene, with certainty 0 (possibly), 1 (likely) or 2 (clearly a whale). Water chips
come only from grid cells the annotator marked as fully scanned. The chips were made with:

```bash
# 1. Cut 512 px chips (reads only the needed windows; mount the drive read-only)
uv run --group data python scripts/build_skysat_chips.py --mode extract \
  --scenes-root "<drive>/SatWhale/data/usa-hawaii_west_lahaina_c" \
  --annotations data/skysat_annotations --exclude-chips data/whale/data/images \
  --out data/skysat_chips --water-per-scene 30 --seed 0
# 2. Center-crop to 500 px and split off the uncertain whales
uv run --group data python scripts/crop_and_split_chips.py data/skysat_chips data/skysat_chips_500
# 3. Imaging conditions per chip
uv run --group data python scripts/describe_chip_conditions.py \
  data/skysat_chips_500/main data/skysat_annotations/scene_metadata
# 4. Quality checks; --apply moves duplicates and suspect water chips to set_aside/
uv run --group data python scripts/qc_skysat_chips.py data/skysat_chips_500 \
  --original data/whale/data/images --apply
```

`data/skysat_annotations/` is a local copy of the drive's shapefiles and scene metadata.

## Running it

1. **Get the data.** Run `notebooks/data_ingester.ipynb` (Colab), or copy `data/` from a teammate.
2. **Get the SatViT weights (optional).** Download `SatViT_V2.pt` into `checkpoints/`, using the
   links in `satvit/README.md`. Without the file, the SatViT run is skipped.
3. **Train.** Run `notebooks/train_chip_classifier.ipynb`. It writes `checkpoints/<RUN_NAME>/`:
   `run.json` (datasets, the exact train/val/test chip lists, seed, settings) and one `.pt` per
   model. It reports validation scores only.
4. **Evaluate.** Run `notebooks/evaluate_models.ipynb` on the same `RUN_NAME`. It compares the
   models with a majority-class and a logistic-regression baseline on the test split (used
   once), with group-bootstrap confidence intervals, PR/ROC curves, calibration, error analysis
   by date and weather, failure examples and an ablation table across runs.

**Splits.** `SPLIT_MODE = 'group'` (default) keeps chips from the same day whose footprints
overlap in one split, so the same boat or patch of water can't be in train and test.
`'date'` holds out whole days (new weather), and `'random'` ignores groups (comparison only).

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
│   ├── data_ingester.ipynb          downloads both chip sets from Google Drive
│   ├── train_chip_classifier.ipynb  trains the three models, saves a run folder
│   └── evaluate_models.ipynb        baselines, test metrics, error analysis, ablations
├── src/whaleboat/     shared code: chip loading, splits, model inputs, models, metrics
├── scripts/           build, crop, describe and quality-check the SkySat chips
├── satvit/            SatViT-V2 encoder code (see satvit/README.md for the weights)
├── checkpoints/       model weights and run folders (git-ignored)
└── data/              imagery (git-ignored)
```

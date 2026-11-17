# Whale and Boat Detection in Planet SkySat Imagery

ICS 635 semester project. Team: Shea Stevens, Joseph Behrens and Kim Cates.
Collaborator: Jake Linksy, PhD (HIMB, Madin lab).

This repo finds whales and boats in 500x500 Planet SkySat chips. The main work is **detection**:
a CNN baseline (Faster R-CNN) and a DETR model (Conditional DETR), tuned and trained the same way
on the same scene-grouped split, plus a weather-augmentation ablation. An earlier **chip
classifier** (water / whale / boat per chip: EfficientNet-B0, DeiT-Tiny, SatViT-V2) is kept for
comparison.

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

## Running it: detection

The notebooks run locally or on Google Colab (GPU runtime; checkpoints then go to Google Drive).

1. **Get the data.** Run `notebooks/data_ingester.ipynb`, or copy `data/` from a teammate.
2. **Train the CNN baseline.** Run `notebooks/train_faster_rcnn.ipynb`. It searches
   hyperparameters with Optuna on the validation split, trains the best setting, picks the score
   cutoff that finds `TARGET_RECALL` (90%) of validation whales and boats, and saves
   `checkpoints/frcnn_main/`.
3. **Train DETR.** Run `notebooks/train_detr.ipynb` the same way (`checkpoints/detr_main/`).
4. **Ablation.** Rerun `train_detr.ipynb` with `RUN_NAME = 'detr_noweather'`,
   `WEATHER_AUG = False` and `SPLIT_FROM = PARAMS_FROM = 'detr_main'`.
5. **Evaluate.** Run `notebooks/evaluate_detectors.ipynb`. It scores everything on the test split
   (used once) with group-bootstrap intervals and compares against a non-learned blob detector.
   It also shows PR curves, recall by annotator certainty, results by haze, water roughness and
   sun angle, failure examples, DETR attention maps and the ablation.

**Scoring.** Labels are points, so training turns each point into a fixed-size box (the size is
tuned), and scoring counts a detection as correct when its center is within 10 m of a point.
Missing a whale costs more than a false alarm, so models are compared by average precision. The
operating cutoff is set for recall, and its false alarms per km² are reported.

## Running it: chip classifier

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
│   ├── train_faster_rcnn.ipynb      detection: CNN baseline (tune, train, set cutoff)
│   ├── train_detr.ipynb             detection: Conditional DETR (same steps)
│   ├── evaluate_detectors.ipynb     detection: test results, conditions, attention, ablation
│   ├── train_chip_classifier.ipynb  classifier: trains the three models, saves a run folder
│   └── evaluate_models.ipynb        classifier: baselines, test metrics, error analysis
├── src/whaleboat/     shared code: chips, splits, detection labels/scoring, models, metrics
├── scripts/           build, crop, describe and quality-check the SkySat chips
├── satvit/            SatViT-V2 encoder code (see satvit/README.md for the weights)
├── checkpoints/       model weights and run folders (git-ignored)
└── data/              imagery (git-ignored)
```

# Whale and Boat Detection in Planet SkySat Imagery

ICS 635 semester project. Team: Shea Stevens, Joseph Behrens and Kim Cates.
Collaborator: Jake Linksy, PhD (HIMB, Madin lab).

## Scope

The proposal compared many classical and deep-learning approaches. Following
the professors' feedback, we're narrowing it to **one CNN detector as a
baseline and one DETR-family detector**, with the analysis centered on DETR.

| Model | Implementation | Why |
|---|---|---|
| **CNN baseline:** Faster R-CNN, ResNet-50 FPN | `torchvision`, COCO-pretrained | Already in the proposal, well understood, reads COCO data directly |
| **DETR:** Conditional DETR, ResNet-50 | `transformers` (`microsoft/conditional-detr-resnet-50`) | Fine-tunes much faster than the original DETR and exposes its cross-attention maps |

Both models use a **ResNet-50 backbone**, so differences come from the detection
head (anchors + NMS vs object queries + attention) and not the feature extractor.
If Conditional DETR struggles on small whales, Deformable DETR is the fallback
because it was designed for small objects.

### Questions

1. **How do the two detectors compare on whales and boats**, especially small
   objects? Results are reported with scene-level bootstrap CIs.
2. **What does DETR attend to?** We'll look at decoder cross-attention per
   object query: whale body, wake, glint, or something else? We'll compare it
   with Grad-CAM on the CNN.
3. **Where do they fail?** We'll look at false positives on glare and
   whitecaps, and at results by condition if condition labels exist.

### Cut, or moved to later milestones

- **Later** (if time allows):
  - an SVM probe on DETR query embeddings
  - a boosted-tree re-scorer on detector outputs
  - Optuna tuning
  - NIR deglinting and filtering ablations
- **Cut:** handcrafted-feature baselines (HOG/LBP/GLCM + SVM/RF/XGBoost),
  RetinaNet, YOLO and the stacked ensemble.

## Roles

| Track | Owner | Owns |
|---|---|---|
| Data and evaluation | data lead | Getting chips and COCO labels off the drive into shared storage, splits (`DATA.md`), condition tags |
| CNN baseline | CNN lead | Faster R-CNN fine-tuning, Grad-CAM, CNN results section |
| DETR | Shea | Conditional DETR fine-tuning, attention analysis, DETR results section |

Everyone writes up their own track. The **shared contract** keeps the results
comparable:

- the same `data/splits/*.json`
- predictions saved as COCO results json at `results/<model>/preds_test.json`.
  `results/` isn't committed, so copy each model's folder to the shared drive
  for comparisons.
- scoring only through `whaleboat.eval`

## Milestone 2: Wed Oct 28 (runnable prototype + early results)

- [ ] **Data:** a prototype subset (a few hundred to ~2k chips, empty chips included) in shared storage, plus `all.json` and splits
- [ ] **CNN:** Faster R-CNN fine-tunes end to end and writes `preds_test.json`
- [ ] **DETR:** Conditional DETR fine-tunes end to end and writes `preds_test.json`
- [ ] Comparison table from `whaleboat.eval` with bootstrap CIs
- [ ] A few example predictions from each model, including failures
- [ ] One DETR attention map, to preview the main analysis
- [ ] Next steps

The CNN and DETR tracks don't need to wait for data. Get the training loop
running on any small COCO-format sample first, then swap in the real splits.

Early numbers can be bad. What matters is that everything runs end to end and
the evaluation is honest.

## Setup

```bash
uv sync
uv run pre-commit install   # once per clone
uv run jupyter lab
```

Notebooks go in `notebooks/cnn/` and `notebooks/detr/`. Shared code is in
`src/whaleboat/`. Use `import whaleboat` instead of copying it into notebooks.

Until the real data arrives, generate fake chips in the same format:

```bash
uv run python -m whaleboat.synthetic data/synthetic
```

`notebooks/detr/conditional_detr.ipynb` runs end to end on these chips:
training, val mAP, test predictions, eval and cross-attention plots. Scores on
fake data mean nothing; the point is that the pipeline works.

### Code style

[Ruff](https://docs.astral.sh/ruff/) enforces PEP 8 style (100-character
lines), catches unused imports and undefined names, and sorts imports.

- The **pre-commit hook** runs ruff and strips notebook outputs on every commit.
  If it changes a file, the commit stops; `git add` the fixes and commit again.
- **GitHub Actions** runs the same ruff checks on every push and pull request.
- To run it by hand: `uv run ruff check --fix . && uv run ruff format .`

## Evaluation

```bash
uv run python -m whaleboat.eval data/splits/test.json \
    results/cnn/preds_test.json results/detr/preds_test.json \
    --bootstrap 200 --out results/comparison.json
```

This reports COCO AP, AP50, AP_small, AR100 and per-class AP50, each with a
95% CI from resampling whole test scenes. With two models it also reports a
paired CI on the difference, which answers whether the gap between the models
is real.

Prediction format, one entry per box, with `image_id` matching the split json:

```json
[{"image_id": 12, "category_id": 1, "bbox": [x, y, w, h], "score": 0.91}]
```

Keep low-confidence boxes (score ≥ ~0.05, up to 100 per chip). mAP needs the
full precision–recall curve, so don't threshold before saving.

## Layout

```
├── DATA.md            data contract and open questions about the dataset
├── src/whaleboat/
│   ├── splits.py      scene-grouped train/val/test split
│   └── eval.py        COCO mAP + scene-bootstrap CIs
├── notebooks/{cnn,detr}/
├── results/           predictions, configs, metrics (git-ignored; share via the shared drive)
├── checkpoints/       model weights (git-ignored)
└── data/              imagery (git-ignored)
```

# Follow-ups

Ideas from the first detector results (2026-11-16). Validation, before test evaluation:

| | Whale AP (59) | Boat AP (154) | Mean AP |
|---|---|---|---|
| Faster R-CNN | 0.44 | 0.94 | 0.69 |
| Conditional DETR | 0.41 | 0.96 | 0.69 |

Boats work for both; whales are the weak class. Most items below test or address the guess that
we are limited by the number of labeled whales.

## 1. Learning curve by training-set size (do first)

Train Faster R-CNN on 25%, 50% and 75% of the training scenes (the 100% run is `frcnn_main`) and
plot validation whale AP against the fraction.

- Subsample by scene/location group, not by chip, so overlapping chips don't leak.
- Reuse the tuned parameters from `frcnn_main` (`PARAMS_FROM`); no new search.
- Still rising at 100% → more labeled whales is the clearest next step. Flat → the limit is whale
  difficulty or label quality, not quantity.
- Cost: three final runs, about 1-2 h each locally; faster on Colab.

## 2. Tune for whales, not mean AP

The search picked settings by mean AP, but boat AP is ~0.95 in every trial, so the search mostly
followed noise. Rerun with `SELECTION_METRIC='whale_ap'`. Cost: one search per model (Colab).

## 3. Dropout (DETR)

Conditional DETR already uses dropout 0.1 by default; we never searched it. Add `dropout` in
{0, 0.1, 0.2} to the DETR search space. Faster R-CNN (torchvision) has no dropout; skip it there.
Expected gain is small: regularization helps generalize from the data we have, it doesn't add
whale variety. Cost: one DETR search, ~8-10 h locally, so run on Colab.

## 4. Monte Carlo dropout (uncertainty)

Keep dropout on at prediction time, predict each chip several times, and use how consistently a
detection appears as a confidence signal. Useful for ranking or flagging detections for the
researchers' review. Separate, nice-to-have.

## 5. Copy-paste whale augmentation

Paste whale crops onto empty water chips. Only worth trying if the weather on/off ablation shows
augmentation helps, and after reading the failure examples (bad labels can't be augmented away).
Risk: the model learns paste artifacts; validation/test stay real images, so scores can't be
inflated, only wasted effort.

## 6. RGB + NIR input (teammate)

Feed the 4th (near-infrared) band by widening the first conv layer, NIR weights initialized from
red. Surfaced whales and boats are bright against near-black water in NIR; submerged whales are
invisible in NIR (absorbed within the top few tens of cm), so report whales and boats separately.
Run as an ablation on Faster R-CNN.

## Related: SatViT classifier comparison (teammate)

SatViT-V2 is pretrained at 10 m with 15 bands; SkySat is 0.5 m with 4. Expect it to do no better
than DeiT-Tiny / EfficientNet-B0, but it is a fair "does satellite pretraining help despite the
mismatch?" comparison. To put it beside the detectors, score detectors at chip level (max whale /
boat score per chip) on the **same scene-grouped test chips**; the old random split leaks.

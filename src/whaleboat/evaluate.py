"""Evaluation helpers: predictions, decision rule, metrics with group-bootstrap CIs, baselines."""

import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support

from .data import CLASSES, make_loader, read_chip
from .models import MODELS, build_model

WHALE = CLASSES.index('whale')


# ---------------------------------------------------------------- runs and predictions


def load_run(run_dir):
    return json.loads((Path(run_dir) / 'run.json').read_text())


def load_trained_model(run_dir, name, satvit_dir, device):
    ckpt = torch.load(Path(run_dir) / f'{name}.pt', map_location='cpu', weights_only=False)
    model = build_model(name, len(CLASSES), satvit_dir=satvit_dir, pretrained=False)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def predict_probs(model, name, chips, band_lo, band_hi, device, workers=2):
    """Softmax probabilities, shape (n, n_classes), in the order of `chips`."""
    cfg = MODELS[name]
    loader = make_loader(
        chips,
        cfg['img_size'],
        name == 'satvit',
        band_lo,
        band_hi,
        cfg['batch_size'] * 2,
        workers=workers,
    )
    out = []
    for x, _, _ in loader:
        out.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


# ---------------------------------------------------------------- decision rule


def decide(probs, whale_threshold=None):
    """argmax, except that whale is predicted whenever p(whale) >= whale_threshold."""
    pred = probs.argmax(1)
    if whale_threshold is not None:
        is_whale = probs[:, WHALE] >= whale_threshold
        other = probs.copy()
        other[:, WHALE] = -1
        pred = np.where(is_whale, WHALE, other.argmax(1))
    return pred


def pick_whale_threshold(probs, y, grid=np.linspace(0.05, 0.95, 91)):
    """Threshold maximizing macro F1 on the data given (use validation data only)."""
    scores = [f1_score(y, decide(probs, t), average='macro', zero_division=0) for t in grid]
    best = int(np.argmax(scores))
    return float(grid[best]), float(scores[best])


# ---------------------------------------------------------------- metrics


def metrics(y, pred, probs=None):
    labels = list(range(len(CLASSES)))
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    out = {
        'n': int(len(y)),
        'accuracy': float(np.mean(y == pred)),
        'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
    }
    for i, c in enumerate(CLASSES):
        out[f'{c}_precision'], out[f'{c}_recall'], out[f'{c}_f1'] = (
            float(p[i]),
            float(r[i]),
            float(f[i]),
        )
    if probs is not None and (y == WHALE).any() and (y != WHALE).any():
        out['whale_ap'] = float(average_precision_score(y == WHALE, probs[:, WHALE]))
    return out


def group_bootstrap(y, pred, groups, probs=None, keys=('macro_f1', 'whale_recall'), n=1000, seed=0):
    """95% CI by resampling whole groups (dates/locations) with replacement.

    Chips in one group are correlated (same day, same conditions), so resampling chips would
    understate the uncertainty. With few test groups the interval is wide, and that's honest.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    samples = {k: [] for k in keys}
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        m = metrics(y[idx], pred[idx], None if probs is None else probs[idx])
        for k in keys:
            if k in m:
                samples[k].append(m[k])
    return {
        k: tuple(np.percentile(v, [2.5, 97.5])) if v else (np.nan, np.nan)
        for k, v in samples.items()
    }


def expected_calibration_error(probs, y, bins=10):
    """ECE of the top-class confidence (equal-width bins)."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ---------------------------------------------------------------- baseline features


def chip_features(path):
    """Hand-made per-chip statistics for the logistic-regression baseline.

    Per band: mean, std, 99th percentile (bright objects), all relative to the chip's own water
    via log scaling; plus the strongest local brightness anomaly (a whale or boat is a small,
    distinct blob) and the number of anomaly blobs (whitecaps make many).
    """
    a = read_chip(path).astype(np.float32)
    valid = ~(a == 0).all(0)
    v = a[:, valid]
    feats = []
    for band in v:
        feats += [np.log1p(band.mean()), np.log1p(band.std()), np.log1p(np.percentile(band, 99))]
    b = a.mean(0)
    b[~valid] = np.median(b[valid])
    resid = ndimage.uniform_filter(b, 5) - ndimage.gaussian_filter(b, 15)
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826 + 1e-6
    z = np.abs(resid) / mad
    feats += [np.log1p(z.max()), np.log1p(ndimage.label(z > 30)[1]), valid.mean()]
    return np.array(feats, dtype=np.float32)

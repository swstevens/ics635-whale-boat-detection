"""Detection data and scoring, shared by both detectors so they are trained and scored the same way.

Labels are points (one per whale or boat), not boxes. For training, each point becomes a square
box of a fixed size per class (a modelling choice, tuned). For scoring, a detection counts as
correct if its center is within MATCH_RADIUS_M of an annotated point of the same class, so the
invented box size never affects the score.

Whales the annotator marked "possibly" (certainty 0) are left out of training targets and are
"ignore" points in scoring: a detection on one is neither a hit nor a false alarm, and missing one
does not lower recall. They are scored separately (recall by certainty).
"""

import csv
import math
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import RGB_BANDS, load_chips, load_skysat, read_chip

DET_CLASSES = ['whale', 'boat']
PIXEL_M = 0.5  # SkySat pansharpened ground sample distance
CHIP_PX = 500
CHIP_KM2 = (CHIP_PX * PIXEL_M) ** 2 / 1e6  # 0.0625 km2 per chip


# ---------------------------------------------------------------- points


def _skysat_points(folder):
    pts = {}
    with open(Path(folder) / 'points.csv', newline='') as f:
        for r in csv.DictReader(f):
            if r['cls'] not in DET_CLASSES:
                continue  # 'poi': points of interest, not a class
            cert = int(r['certainty']) if r['certainty'].strip().isdigit() else None
            pts.setdefault(f'skysat/{r["chip_id"]}', []).append(
                {'cls': r['cls'], 'x': float(r['px']), 'y': float(r['py']), 'certainty': cert}
            )
    return pts


def _mask_points(mask_path, cls, merge_px=6, min_px=20):
    """Object centers from an original-chip mask.

    Boat masks are often split into many fragments (wake, deck, shadow), so fragments within
    merge_px of each other are joined first; specks under min_px pixels are dropped.
    """
    with rasterio.open(mask_path) as src:
        m = src.read().any(0)
    joined = ndimage.binary_dilation(m, iterations=merge_px)
    lab, n = ndimage.label(joined)
    lab = lab * m  # measure only real mask pixels
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(xs) >= min_px:
            out.append({'cls': cls, 'x': float(xs.mean()), 'y': float(ys.mean()), 'certainty': 2})
    return out


def _original_points(orig_dir, chips):
    labels = Path(orig_dir) / 'labels'
    pts = {}
    for c in chips:
        stem = Path(c['path']).name
        for cls, sub in [('whale', 'whales'), ('boat', 'boats')]:
            p = labels / sub / stem
            if p.exists():
                pts.setdefault(c['name'], []).extend(_mask_points(p, cls))
    return pts


def attach_points(chips, orig_dir, skysat_dir, subset='main'):
    """Add chip['points'] (list of {cls, x, y, certainty, ignore}) in chip pixel coordinates."""
    pts = {}
    if any(c['source'] == 'skysat' for c in chips):
        pts.update(_skysat_points(Path(skysat_dir) / subset))
    orig = [c for c in chips if c['source'] == 'original']
    if orig:
        pts.update(_original_points(orig_dir, orig))
    for c in chips:
        c['points'] = [
            {**p, 'ignore': p['cls'] == 'whale' and p['certainty'] == 0}
            for p in pts.get(c['name'], [])
        ]
    return chips


def load_detection_chips(datasets, orig_dir, skysat_dir):
    return attach_points(load_chips(datasets, orig_dir, skysat_dir), orig_dir, skysat_dir)


def load_uncertain_whales(orig_dir, skysat_dir):
    """The 'possibly a whale' chips (never trained on). Their whales are scored, not ignored."""
    chips = attach_points(
        load_skysat(skysat_dir, 'uncertain_whales'), orig_dir, skysat_dir, subset='uncertain_whales'
    )
    for c in chips:
        for p in c['points']:
            p['ignore'] = False
    return chips


def count_points(chips):
    out = {}
    for c in chips:
        for p in c['points']:
            key = f'{p["cls"]} (possibly)' if p['ignore'] else p['cls']
            out[key] = out.get(key, 0) + 1
    return out


# ---------------------------------------------------------------- image cache


def stretch_to_uint8(chip, band_lo, band_hi):
    """(4, H, W) uint16 -> (3, H, W) uint8 RGB with the fixed per-band training stretch."""
    lo = np.asarray(band_lo, np.float32)[RGB_BANDS, None, None]
    hi = np.asarray(band_hi, np.float32)[RGB_BANDS, None, None]
    rgb = (chip[RGB_BANDS].astype(np.float32) - lo) / (hi - lo)
    return (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)


class ChipCache(dict):
    """name -> (3, 500, 500) uint8 RGB, read once into memory (~0.75 MB per chip).

    Hyperparameter search trains many short models; re-reading GeoTIFFs every epoch would make
    the disk, not the GPU, the bottleneck. NIR is not used: both detectors start from
    COCO-pretrained RGB weights.
    """

    def __init__(self, chips, band_lo, band_hi, workers=8):
        super().__init__()
        todo = [c for c in chips if c['name'] not in self]

        def load(c):
            return c['name'], stretch_to_uint8(read_chip(c['path']), band_lo, band_hi)

        with ThreadPoolExecutor(workers) as ex:
            for name, img in ex.map(load, todo):
                self[name] = img

    def add(self, chips, band_lo, band_hi):
        self.update(ChipCache([c for c in chips if c['name'] not in self], band_lo, band_hi))


# ---------------------------------------------------------------- augmentation


def _rot90_points(xy, size, k):
    """Point coordinates matching torch.rot90(img, k, dims=[1, 2]) on a size x size image."""
    for _ in range(k % 4):
        xy = np.stack([xy[:, 1], size - xy[:, 0]], 1)
    return xy


def geometric_aug(img, xy):
    """Random flips and 90-degree rotations (overhead imagery has no 'up')."""
    size = img.shape[-1]
    xy = xy.copy()
    if random.random() < 0.5:
        img = img.flip(-1)
        xy[:, 0] = size - xy[:, 0]
    if random.random() < 0.5:
        img = img.flip(-2)
        xy[:, 1] = size - xy[:, 1]
    k = random.randint(0, 3)
    img = torch.rot90(img, k, dims=[1, 2])
    return img, _rot90_points(xy, size, k)


def _smooth_field(h, w, cells=4):
    """Random low-frequency field in [0, 1], e.g. patchy haze or sun glint."""
    f = torch.rand(1, 1, cells, cells)
    f = F.interpolate(f, size=(h, w), mode='bicubic', align_corners=False)[0]
    f = f - f.min()
    return f / (f.max() + 1e-6)


def weather_aug(img, p=0.5):
    """Photometric changes that mimic conditions across SkySat captures. img: (3, H, W) in [0, 1].

    - haze: blend toward a bright, slightly blue veil, uniform or patchy
    - sun glint: bright low-frequency patches added to the water
    - exposure: brightness, contrast and gamma (sun elevation, view angle)
    - color cast: per-channel gain (water color, atmosphere)
    - sensor noise and slight blur
    Each is applied independently with probability p. This is the only thing switched off in the
    weather-augmentation ablation.
    """
    _, h, w = img.shape
    if random.random() < p:  # haze
        veil = torch.tensor([0.80, 0.82, 0.88]).view(3, 1, 1) * random.uniform(0.85, 1.1)
        a = random.uniform(0.1, 0.45)
        amount = a * (_smooth_field(h, w) if random.random() < 0.5 else torch.ones(1, h, w))
        img = img * (1 - amount) + veil * amount
    if random.random() < p:  # sun glint
        img = img + random.uniform(0.05, 0.35) * _smooth_field(h, w, cells=6) ** 3
    if random.random() < p:  # exposure
        img = (img - 0.5) * random.uniform(0.7, 1.3) + 0.5 + random.uniform(-0.1, 0.1)
        img = img.clamp(0, 1) ** random.uniform(0.75, 1.33)
    if random.random() < p:  # color cast
        img = img * torch.empty(3, 1, 1).uniform_(0.9, 1.1)
    if random.random() < p:  # noise, sometimes after a slight blur
        if random.random() < 0.5:
            k = torch.tensor([0.25, 0.5, 0.25])
            img = F.conv2d(
                F.pad(img[None], (1, 1, 1, 1), mode='replicate'),
                (k[:, None] * k[None, :]).expand(3, 1, 3, 3),
                groups=3,
            )[0]
        img = img + torch.randn_like(img) * random.uniform(0.005, 0.03)
    return img.clamp(0, 1)


# ---------------------------------------------------------------- dataset


class DetectionDataset(Dataset):
    """Returns (image (3, S, S) float in [0, 1], target) where S = round(500 * scale).

    target: boxes (N, 4) xyxy in S-pixel coordinates, labels (N,) index into DET_CLASSES, name.
    Boxes are squares of box_px[cls] chip pixels around each training point; ignore points
    (possible whales) are left out.
    """

    def __init__(
        self, chips, cache, scale=1.0, box_px=None, train=False, weather=False, weather_p=0.5
    ):
        self.chips = list(chips)
        self.cache = cache
        self.scale = scale
        self.size = round(CHIP_PX * scale)
        self.box_px = box_px or {'whale': 40, 'boat': 40}
        self.train = train
        self.weather = weather
        self.weather_p = weather_p

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, i):
        c = self.chips[i]
        img = torch.from_numpy(self.cache[c['name']]).float() / 255
        if self.size != CHIP_PX:
            img = F.interpolate(
                img[None], size=(self.size, self.size), mode='bilinear', antialias=True
            )[0]
        pts = [p for p in c['points'] if not p['ignore']]
        xy = np.array([[p['x'], p['y']] for p in pts], np.float32).reshape(-1, 2) * self.scale
        cls = [p['cls'] for p in pts]
        if self.train:
            img, xy = geometric_aug(img, xy)
            if self.weather:
                img = weather_aug(img, self.weather_p)
        half = np.array([self.box_px[k] for k in cls], np.float32) * self.scale / 2
        boxes = np.stack([xy[:, 0] - half, xy[:, 1] - half, xy[:, 0] + half, xy[:, 1] + half], 1)
        boxes = np.clip(boxes, 0, self.size).reshape(-1, 4)
        target = {
            'boxes': torch.from_numpy(boxes).float(),
            'labels': torch.tensor([DET_CLASSES.index(k) for k in cls], dtype=torch.long),
            'name': c['name'],
        }
        return img.contiguous(), target


def det_collate(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs), list(targets)


def make_det_loader(
    chips,
    cache,
    batch_size,
    scale=1.0,
    box_px=None,
    train=False,
    weather=False,
    water_frac=None,
    workers=2,
    seed=0,
):
    """water_frac: share of water-only chips per epoch when training (None = as in the data).

    Water chips carry no boxes but teach the model what to leave alone (whitecaps, glint).
    """
    ds = DetectionDataset(chips, cache, scale, box_px, train=train, weather=weather)
    sampler = None
    if train and water_frac is not None:
        is_water = np.array([not any(not p['ignore'] for p in c['points']) for c in ds.chips])
        n_w, n_o = is_water.sum(), (~is_water).sum()
        w = np.where(is_water, water_frac / max(n_w, 1), (1 - water_frac) / max(n_o, 1))
        sampler = WeightedRandomSampler(
            torch.as_tensor(w, dtype=torch.double),
            len(ds),
            generator=torch.Generator().manual_seed(seed),
        )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=workers,
        collate_fn=det_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


# ---------------------------------------------------------------- matching and metrics


def match_chip(dets, points, radius_px):
    """Greedy point matching for one chip and class, highest score first.

    dets: list of (x, y, score). points: list of point dicts (same class).
    Returns (det_status, point_score): det_status per detection is 1 (hit), 0 (false alarm) or
    -1 (on an ignore point: not counted); point_score is the score of the detection that found
    each point, or -inf if none did.
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i][2])
    status = [0] * len(dets)
    pscore = [-math.inf] * len(points)
    taken = [False] * len(points)
    pxy = np.array([[p['x'], p['y']] for p in points], np.float32).reshape(-1, 2)
    for i in order:
        x, y, s = dets[i]
        if not len(pxy):
            break
        d = np.hypot(pxy[:, 0] - x, pxy[:, 1] - y)
        best, best_d = None, radius_px
        for j in np.argsort(d):
            if d[j] > radius_px:
                break
            if not taken[j]:
                best = j
                break
        if best is None:
            # A second detection on an already-found object is a false alarm, except near an
            # ignore point, where nothing counts.
            near_ignore = any(points[j]['ignore'] and d[j] <= best_d for j in range(len(points)))
            status[i] = -1 if near_ignore else 0
            continue
        taken[best] = True
        if points[best]['ignore']:
            status[i] = -1
        else:
            status[i] = 1
            pscore[best] = s
    return status, pscore


def evaluate_detections(preds, chips, radius_px, classes=DET_CLASSES):
    """Pool matches over chips, per class.

    preds: name -> list of {'cls', 'x', 'y', 'score'} in chip pixels.
    Returns class -> dict(scores, hit (bool), det_chip, det_xy, point_score, points (meta),
    n_points, n_chips). Detections on ignore points are dropped.
    """
    out = {}
    for k in classes:
        scores, hits, det_chip, det_xy, pscores, pmeta = [], [], [], [], [], []
        for c in chips:
            dets = [(d['x'], d['y'], d['score']) for d in preds.get(c['name'], []) if d['cls'] == k]
            pts = [p for p in c['points'] if p['cls'] == k]
            status, ps = match_chip(dets, pts, radius_px)
            for (x, y, s), st in zip(dets, status):
                if st >= 0:
                    scores.append(s)
                    hits.append(st == 1)
                    det_chip.append(c['name'])
                    det_xy.append((x, y))
            for p, s in zip(pts, ps):
                if not p['ignore']:
                    pscores.append(s)
                    pmeta.append({**p, 'chip': c['name']})
        out[k] = {
            'scores': np.array(scores, np.float64),
            'hit': np.array(hits, bool),
            'det_chip': np.array(det_chip, object),
            'det_xy': np.array(det_xy, np.float32).reshape(-1, 2),
            'point_score': np.array(pscores, np.float64),
            'points': pmeta,
            'n_points': len(pscores),
            'n_chips': len(chips),
        }
    return out


def pr_curve(res):
    """Precision and recall at every score cutoff, highest score first."""
    order = np.argsort(-res['scores'], kind='stable')
    hit = res['hit'][order]
    tp = np.cumsum(hit)
    fp = np.cumsum(~hit)
    recall = tp / max(res['n_points'], 1)
    precision = tp / np.maximum(tp + fp, 1)
    return precision, recall, res['scores'][order]


def average_precision(res):
    """Area under the precision-recall curve (all-point, precision made non-increasing)."""
    if res['n_points'] == 0:
        return float('nan')
    precision, recall, _ = pr_curve(res)
    if not len(precision):
        return 0.0
    p = np.concatenate([[0.0], precision, [0.0]])
    r = np.concatenate([[0.0], recall, [recall[-1]]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


def threshold_for_recall(res, target):
    """Highest score cutoff that still finds `target` of the points (pick on validation only).

    If the target can't be reached (some points never found), returns the lowest score seen so
    every detection is kept.
    """
    found = np.sort(res['point_score'][np.isfinite(res['point_score'])])[::-1]
    need = math.ceil(target * res['n_points'])
    if need == 0:
        return float('inf')
    if need <= len(found):
        return float(found[need - 1])
    return float(res['scores'].min()) if len(res['scores']) else 0.0


def _cutoff_curve(res):
    """Every distinct cutoff (high to low) with the hits and false alarms kept at score >= it."""
    order = np.argsort(-res['scores'], kind='stable')
    s = res['scores'][order]
    hit = res['hit'][order]
    tp, fp = np.cumsum(hit), np.cumsum(~hit)
    last = np.r_[s[1:] != s[:-1], True]  # ties: a cutoff keeps all detections with that score
    return s[last], tp[last], fp[last]


def threshold_fbeta(res, beta=2.0):
    """Cutoff maximizing F-beta (pick on validation only). beta=2 weighs recall 4x precision:
    the standard choice when a missed object costs more than a false alarm."""
    s, tp, fp = _cutoff_curve(res)
    if not len(s) or res['n_points'] == 0:
        return float('inf')
    recall, precision = tp / res['n_points'], tp / (tp + fp)
    b2 = beta**2
    f = (1 + b2) * precision * recall / np.maximum(b2 * precision + recall, 1e-12)
    return float(s[int(np.argmax(f))])


def threshold_for_fa_budget(res, max_fa_per_km2):
    """Lowest cutoff whose false alarms stay within the budget (pick on validation only):
    the most whales a reviewer can be shown for a fixed amount of checking."""
    s, _, fp = _cutoff_curve(res)
    ok = np.flatnonzero(fp <= max_fa_per_km2 * res['n_chips'] * CHIP_KM2)
    return float(s[ok[-1]]) if len(ok) else float('inf')


def threshold_for_count(res):
    """Cutoff where the number of detections kept is closest to the number of objects (pick on
    validation only). Misses and false alarms then cancel in the total, which is what a count
    needs; individual detections can still be wrong."""
    s, tp, fp = _cutoff_curve(res)
    if not len(s) or res['n_points'] == 0:
        return float('inf')
    return float(s[int(np.argmin(np.abs(tp + fp - res['n_points'])))])


def counts_by(res, thr, chips, group_of):
    """{group: (true count, predicted count at score >= thr, sum of all detection scores)}.

    Possible whales are left out of both the true and the predicted counts (detections on them
    are dropped in scoring). group_of maps a chip name to a key such as its capture.
    """
    out = {group_of(c['name']): [0, 0, 0.0] for c in chips}
    for p in res['points']:
        out[group_of(p['chip'])][0] += 1
    for name, s in zip(res['det_chip'], res['scores']):
        g = out[group_of(name)]
        g[1] += int(s >= thr)
        g[2] += float(s)
    return {k: tuple(v) for k, v in out.items()}


def at_threshold(res, thr):
    """Recall, precision and false alarms (count, per chip, per km2) at score >= thr."""
    keep = res['scores'] >= thr
    tp = int(res['hit'][keep].sum())
    fp = int((~res['hit'][keep]).sum())
    found = int((res['point_score'] >= thr).sum())
    return {
        'recall': found / res['n_points'] if res['n_points'] else float('nan'),
        'precision': tp / (tp + fp) if tp + fp else float('nan'),
        'false_alarms': fp,
        'fa_per_chip': fp / max(res['n_chips'], 1),
        'fa_per_km2': fp / max(res['n_chips'] * CHIP_KM2, 1e-9),
    }


def summarize(results, thresholds=None):
    """One flat dict: <cls>_ap, and with thresholds <cls>_recall / _precision / _fa_per_km2."""
    out = {}
    for k, res in results.items():
        out[f'{k}_ap'] = average_precision(res)
        out[f'{k}_n'] = res['n_points']
        if thresholds and k in thresholds:
            for m, v in at_threshold(res, thresholds[k]).items():
                out[f'{k}_{m}'] = v
    aps = [out[f'{k}_ap'] for k in results if not math.isnan(out[f'{k}_ap'])]
    out['mean_ap'] = float(np.mean(aps)) if aps else float('nan')
    return out


def group_bootstrap(preds, chips, radius_px, thresholds=None, n=500, seed=0, classes=DET_CLASSES):
    """95% CIs for summarize() by resampling whole groups of chips (same day, overlapping).

    Returns {metric: (lo, hi)}.
    """
    rng = np.random.default_rng(seed)
    by_group = {}
    for c in chips:
        by_group.setdefault(c['group'], []).append(c)
    groups = list(by_group)
    samples = {}
    for _ in range(n):
        sample, sp = [], {}
        for b, g in enumerate(rng.choice(len(groups), len(groups))):
            for c in by_group[groups[g]]:
                key = f'{c["name"]}#{b}'
                sample.append({**c, 'name': key})
                sp[key] = preds.get(c['name'], [])
        m = summarize(evaluate_detections(sp, sample, radius_px, classes), thresholds)
        for k, v in m.items():
            samples.setdefault(k, []).append(v)
    return {
        k: tuple(np.nanpercentile(v, [2.5, 97.5])) if np.isfinite(v).any() else (np.nan, np.nan)
        for k, v in samples.items()
    }


def merge_classes(preds, radius_px):
    """Class-agnostic view: every detection becomes 'object'; near-duplicates (a whale and a boat
    detection on the same spot) keep only the higher score."""
    out = {}
    for name, dets in preds.items():
        kept = []
        for d in sorted(dets, key=lambda d: -d['score']):
            if all(math.hypot(d['x'] - k['x'], d['y'] - k['y']) > radius_px for k in kept):
                kept.append({**d, 'cls': 'object'})
        out[name] = kept
    return out


def agnostic_chips(chips):
    return [{**c, 'points': [{**p, 'cls': 'object'} for p in c['points']]} for c in chips]


# ---------------------------------------------------------------- non-learned baseline


def anomaly_detections(chip, z_min=6.0, max_dets=50):
    """Bright/dark blob detector, no training: local brightness residual in robust z units.

    Whales and boats are small, distinct blobs against water; each connected region above z_min
    becomes one class-agnostic detection scored by its peak z. Whitecaps and glint also fire,
    which is what the learned detectors must do better than.
    """
    a = chip.astype(np.float32)
    valid = ~(a == 0).all(0)
    b = a.mean(0)
    b[~valid] = np.median(b[valid]) if valid.any() else 0
    resid = ndimage.uniform_filter(b, 5) - ndimage.gaussian_filter(b, 15)
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826 + 1e-6
    z = np.abs(resid) / mad
    z[~valid] = 0
    lab, n = ndimage.label(z > z_min)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    peaks = ndimage.maximum(z, lab, idx)
    centers = ndimage.center_of_mass(z, lab, idx)
    dets = [
        {'cls': 'object', 'x': float(cx), 'y': float(cy), 'score': float(s)}
        for (cy, cx), s in zip(centers, peaks)
    ]
    return sorted(dets, key=lambda d: -d['score'])[:max_dets]

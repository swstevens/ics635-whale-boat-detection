"""Chip loading, grouped splits and model inputs, shared by the training and evaluation notebooks.

A chip is a dict: name ('original/<stem>' or 'skysat/<chip_id>'), path, label (class name),
source ('original' or 'skysat'), bounds (UTM footprint), and two grouping keys:

- group: chips that must stay in one split. Chips from the same day whose footprints overlap
  (directly or through a chain) share a group, so the same boat or patch of water imaged in
  back-to-back captures can't sit in train and test at once.
- date: the acquisition day. SPLIT_MODE 'date' holds out whole days (new weather / conditions).

The original 144 chips have no dates; they overlap heavily, so their overlap groups (4 large
locations) serve as both keys.
"""

import csv
import random
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

CLASSES = ['water', 'whale', 'boat']
RGB_BANDS = [2, 1, 0]  # chips store Blue, Green, Red, NIR
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SATVIT_CHANNELS = 15
SATVIT_BAND_SLOTS = [3, 4, 5, 9]  # Sentinel-2 B2, B3, B4, B8 after the 2 Sentinel-1 bands


# ---------------------------------------------------------------- chip tables


def _overlap_groups(bounds):
    parent = list(range(len(bounds)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, a in enumerate(bounds):
        for j in range(i + 1, len(bounds)):
            b = bounds[j]
            if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(bounds))]


def load_original(orig_dir):
    orig_dir = Path(orig_dir)
    chips = []
    for path in sorted((orig_dir / 'images').glob('*.tif')):
        if (orig_dir / 'labels' / 'whales' / path.name).exists():
            label = 'whale'
        elif (orig_dir / 'labels' / 'boats' / path.name).exists():
            label = 'boat'
        else:
            label = 'water'
        with rasterio.open(path) as src:
            bounds = tuple(src.bounds)
        chips.append(
            {'name': f'original/{path.stem}', 'path': path, 'label': label, 'bounds': bounds}
        )
    for chip, g in zip(chips, _overlap_groups([c['bounds'] for c in chips])):
        chip['group'] = chip['date'] = f'original_loc{g}'
        chip['source'] = 'original'
    return chips


def load_skysat(skysat_dir, subset='main'):
    folder = Path(skysat_dir) / subset
    with open(folder / 'manifest.csv', newline='') as f:
        rows = list(csv.DictReader(f))
    chips = [
        {
            'name': f'skysat/{r["chip_id"]}',
            'path': folder / 'images' / f'{r["chip_id"]}.tif',
            # A chip with a whale and a boat counts as whale (the rarer class).
            'label': 'whale' if 'whale' in r['label'] else r['label'],
            'date': f'skysat_{r["date"]}',
            'bounds': tuple(float(r[k]) for k in ['xmin', 'ymin', 'xmax', 'ymax']),
            'source': 'skysat',
        }
        for r in rows
    ]
    by_date = {}
    for c in chips:
        by_date.setdefault(c['date'], []).append(c)
    for date, cs in by_date.items():
        for c, g in zip(cs, _overlap_groups([c['bounds'] for c in cs])):
            c['group'] = f'{date}_area{g}'
    return chips


def overlaps_same_day(chip, others):
    """Whether chip's footprint overlaps any chip in others from the same date."""
    a = chip['bounds']
    return any(
        o['date'] == chip['date']
        and a[0] < o['bounds'][2]
        and o['bounds'][0] < a[2]
        and a[1] < o['bounds'][3]
        and o['bounds'][1] < a[3]
        for o in others
    )


def load_chips(datasets, orig_dir, skysat_dir):
    chips = []
    if 'original' in datasets:
        chips += load_original(orig_dir)
    if 'skysat' in datasets:
        chips += load_skysat(skysat_dir, 'main')
    return chips


def read_chip(path):
    with rasterio.open(path) as src:
        return src.read()  # (4, 500, 500) uint16


# ---------------------------------------------------------------- splits


def _group_split(idx, labels, groups, frac, seed):
    n_groups = len(set(groups[idx]))
    n_splits = min(max(2, round(1 / frac)), n_groups)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest, held = next(sgkf.split(idx, labels[idx], groups[idx]))
    return idx[rest], idx[held]


def make_split(chips, mode='group', val_frac=0.15, test_frac=0.15, seed=42):
    """Return {'train': [...], 'val': [...], 'test': [...]} of chip names.

    'group': overlapping same-day chips stay together (StratifiedGroupKFold keeps class ratios
    roughly even). Tests new places under conditions seen in training.
    'date': whole acquisition days stay together. Tests new days / weather; few groups, so
    splits are lumpy.
    'random': stratified random split that ignores groups (leaks overlapping chips).
    """
    labels = np.array([CLASSES.index(c['label']) for c in chips])
    key = 'date' if mode == 'date' else 'group'
    groups = np.array([c[key] for c in chips])
    idx = np.arange(len(chips))
    if mode in ('group', 'date'):
        trainval, test = _group_split(idx, labels, groups, test_frac, seed)
        train, val = _group_split(trainval, labels, groups, val_frac / (1 - test_frac), seed)
        assert not set(groups[train]) & set(groups[test]), 'a group is in train and test'
        assert not set(groups[train]) & set(groups[val]), 'a group is in train and val'
    elif mode == 'random':
        trainval, test = train_test_split(
            idx, test_size=test_frac, stratify=labels, random_state=seed
        )
        train, val = train_test_split(
            trainval,
            test_size=val_frac / (1 - test_frac),
            stratify=labels[trainval],
            random_state=seed,
        )
    else:
        raise ValueError(f'unknown split mode {mode!r}')
    names = np.array([c['name'] for c in chips])
    return {'train': list(names[train]), 'val': list(names[val]), 'test': list(names[test])}


def describe_split(chips, split):
    by_name = {c['name']: c for c in chips}
    for part, names in split.items():
        cs = [by_name[n] for n in names]
        counts = '  '.join(f'{k}={sum(c["label"] == k for c in cs)}' for k in CLASSES)
        srcs = ' '.join(
            f'{s}={sum(c["source"] == s for c in cs)}' for s in sorted({c['source'] for c in cs})
        )
        groups = len({c['group'] for c in cs})
        dates = len({c['date'] for c in cs})
        print(f'{part:5s} {len(cs):4d}   {counts}   groups={groups} dates={dates}   ({srcs})')


# ---------------------------------------------------------------- model inputs


def band_stretch(paths, seed=42, max_chips=300):
    """Per-band 2nd/98th percentiles from (up to max_chips of) the given training chips."""
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(paths), min(max_chips, len(paths)), replace=False)
    px = np.stack([read_chip(paths[i])[:, ::4, ::4] for i in pick]).astype(np.float32)
    lo = np.percentile(px, 2, axis=(0, 2, 3))
    hi = np.percentile(px, 98, axis=(0, 2, 3))
    return lo.tolist(), hi.tolist()


def to_model_input(chip, img_size, satvit, band_lo, band_hi):
    """(4, H, W) uint16 chip -> model input tensor.

    CNN / ViT: fixed per-band stretch from training chips, RGB, ImageNet normalization.
    SatViT: per-chip, per-band z-score; our 4 bands go into the Sentinel-2 B2/B3/B4/B8 slots and
    the other 11 of its 15 channels stay 0.
    """
    img = torch.from_numpy(chip.astype(np.float32))
    if satvit:
        mean = img.mean(dim=(1, 2), keepdim=True)
        std = img.std(dim=(1, 2), keepdim=True)
        img = (img - mean) / (std + 1e-8)
    else:
        lo = torch.tensor(band_lo, dtype=torch.float32).view(4, 1, 1)
        hi = torch.tensor(band_hi, dtype=torch.float32).view(4, 1, 1)
        img = ((img - lo) / (hi - lo)).clamp(0, 1)
        img = (img[RGB_BANDS] - IMAGENET_MEAN) / IMAGENET_STD
    size = (img_size, img_size)
    img = F.interpolate(img[None], size=size, mode='bilinear', antialias=True)[0]
    if satvit:
        x = torch.zeros(SATVIT_CHANNELS, *size)
        x[SATVIT_BAND_SLOTS] = img
        return x
    return img


class ChipDataset(Dataset):
    """Items are chip dicts; returns (input tensor, label index, chip name)."""

    def __init__(self, chips, img_size, satvit, band_lo, band_hi, train=False):
        self.chips = list(chips)
        self.kw = {'img_size': img_size, 'satvit': satvit, 'band_lo': band_lo, 'band_hi': band_hi}
        self.train = train

    def __len__(self):
        return len(self.chips)

    def __getitem__(self, i):
        c = self.chips[i]
        x = to_model_input(read_chip(c['path']), **self.kw)
        if self.train:  # overhead imagery has no "up": random flips and 90-degree rotations
            if random.random() > 0.5:
                x = torch.flip(x, dims=[2])
            if random.random() > 0.5:
                x = torch.flip(x, dims=[1])
            x = torch.rot90(x, random.randint(0, 3), dims=[1, 2])
        return x, CLASSES.index(c['label']), c['name']


def collate(batch):
    imgs, labels, names = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels), list(names)


def make_loader(chips, img_size, satvit, band_lo, band_hi, batch_size, train=False, workers=2):
    return DataLoader(
        ChipDataset(chips, img_size, satvit, band_lo, band_hi, train=train),
        batch_size=batch_size,
        shuffle=train,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

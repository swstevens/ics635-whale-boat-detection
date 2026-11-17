"""Center-crop chips (512 -> 500, keeping 0.5 m pixels) and split off uncertain-whale chips.

Cropping, not resampling, keeps the pixel size equal to the original 500x500 dataset.
Points that land in the trimmed border are dropped and each chip's label is recomputed.

Chips whose whales are all certainty 0 / blank ("possibly a whale") go to <out>/uncertain_whales,
everything else (water, boat, whale with certainty >= 1) to <out>/main. Each folder gets its own
images/, manifest.csv and points.csv.

Run: uv run --group data python scripts/crop_and_split_chips.py \
       data/skysat_chips data/skysat_chips_500 --size 500
"""

import argparse
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.windows import Window


def label_of(n_whale, n_boat):
    if n_whale and n_boat:
        return 'whale+boat'
    return 'whale' if n_whale else ('boat' if n_boat else 'water')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--size', type=int, default=500)
    args = ap.parse_args()

    man = pd.read_csv(args.src / 'manifest.csv')
    pts = pd.read_csv(args.src / 'points.csv', dtype={'certainty': str})
    with rasterio.open(args.src / 'images' / f'{man.chip_id.iloc[0]}.tif') as s:
        old = s.width
    off = (old - args.size) // 2

    # Shift points into the cropped frame and drop those in the trimmed border.
    pts['px'] -= off
    pts['py'] -= off
    inside = pts.px.between(0, args.size, inclusive='left') & pts.py.between(
        0, args.size, inclusive='left'
    )
    print(f'dropped {(~inside).sum()} points in the {off}px border')
    pts = pts[inside].copy()

    # Recompute per-chip counts, label and certainty from the remaining points.
    def cert(s):
        v = pd.to_numeric(s, errors='coerce')
        return v.max() if v.notna().any() else pd.NA

    g = pts.groupby('chip_id')
    counts = pd.DataFrame(
        {
            'n_whale': g.cls.apply(lambda c: (c == 'whale').sum()),
            'n_boat': g.cls.apply(lambda c: (c == 'boat').sum()),
            'n_poi': g.cls.apply(lambda c: (c == 'poi').sum()),
            'whale_certainty_max': pts[pts.cls == 'whale'].groupby('chip_id').certainty.apply(cert),
        }
    )
    man = man.drop(columns=['n_whale', 'n_boat', 'n_poi', 'whale_certainty_max']).merge(
        counts, left_on='chip_id', right_index=True, how='left'
    )
    man[['n_whale', 'n_boat', 'n_poi']] = man[['n_whale', 'n_boat', 'n_poi']].fillna(0).astype(int)
    old_label = man.label
    man['label'] = [label_of(w, b) for w, b in zip(man.n_whale, man.n_boat)]
    changed = (old_label != man.label).sum()
    print(f'labels changed by the crop: {changed}')

    # New footprint and offsets.
    man['col_off'] += off
    man['row_off'] += off
    res = (man.xmax - man.xmin) / old
    man['xmin'] += off * res
    man['xmax'] -= off * res
    man['ymin'] += off * res
    man['ymax'] -= off * res

    uncertain = (man.n_whale > 0) & ~(man.whale_certainty_max >= 1).fillna(False)
    for name, sub in [('main', man[~uncertain]), ('uncertain_whales', man[uncertain])]:
        d = args.out / name
        (d / 'images').mkdir(parents=True, exist_ok=True)
        for cid in sub.chip_id:
            with rasterio.open(args.src / 'images' / f'{cid}.tif') as s:
                win = Window(off, off, args.size, args.size)
                profile = s.profile | {
                    'width': args.size,
                    'height': args.size,
                    'predictor': 2,
                    'transform': s.window_transform(win),
                }
                with rasterio.open(d / 'images' / f'{cid}.tif', 'w', **profile) as dst:
                    dst.write(s.read(window=win))
                    dst.descriptions = s.descriptions
        sub.to_csv(d / 'manifest.csv', index=False)
        pts[pts.chip_id.isin(sub.chip_id)].to_csv(d / 'points.csv', index=False)
        print(name, len(sub), sub.label.value_counts().to_dict())


if __name__ == '__main__':
    main()

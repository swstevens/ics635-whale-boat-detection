"""Add scene- and chip-level imaging conditions to a chip manifest (local files only).

Scene level (from Planet's <scene>_metadata.json): acquisition time, sun elevation/azimuth,
view angle, and the scene's cloud / haze / clear percentages.
Chip level (from the chip pixels): brightness, NIR spread (roughness: wind chop, whitecaps),
and the share of very bright pixels (whitecaps, sun glint).

Writes <chips>/manifest_conditions.csv. Use it to check that water chips span the same
conditions as whale/boat chips, and to stratify splits by condition.

Run: uv run --group data python scripts/describe_chip_conditions.py \
       data/skysat_chips data/skysat_annotations/scene_metadata
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

SCENE_KEYS = [
    'acquired',
    'sun_elevation',
    'sun_azimuth',
    'view_angle',
    'satellite_azimuth',
    'cloud_percent',
    'light_haze_percent',
    'heavy_haze_percent',
    'clear_percent',
]


def main(chips_dir, meta_dir):
    man = pd.read_csv(chips_dir / 'manifest.csv')

    meta = {}
    for f in meta_dir.glob('*_metadata.json'):
        props = json.loads(f.read_text()).get('properties', {})
        meta[f.name.removesuffix('_metadata.json')] = {k: props.get(k) for k in SCENE_KEYS}
    key = man.scene.map(lambda s: re.match(r'(\d{8}_\d{6}_ssc\d+_u\d+)', s).group(1))
    # A few segments have no metadata file; fall back to a sibling segment of the same capture
    # (same acquisition, so same time / sun / view angle).
    by_capture = {re.sub(r'_u\d+$', '', k): v for k, v in sorted(meta.items())}
    scene_df = pd.DataFrame(
        [meta.get(k) or by_capture.get(re.sub(r'_u\d+$', '', k), {}) for k in key], index=man.index
    )

    rows = []
    for cid in man.chip_id:
        with rasterio.open(chips_dir / 'images' / f'{cid}.tif') as s:
            a = s.read().astype(np.float32)
        valid = ~(a == 0).all(0)
        bgr = a[:3][:, valid]
        nir = a[3][valid]
        bright = bgr.mean(0)
        thresh = np.median(bright) * 2  # "very bright" relative to the chip's own water
        rows.append(
            {
                'mean_rgb': round(float(bright.mean()), 1),
                'nir_mean': round(float(nir.mean()), 1),
                'nir_std': round(float(nir.std()), 1),
                'bright_px_frac': round(float((bright > thresh).mean()), 5),
            }
        )
    out = pd.concat([man, scene_df, pd.DataFrame(rows, index=man.index)], axis=1)
    out['hour_utc'] = pd.to_datetime(out.acquired, format='ISO8601').dt.hour
    out.to_csv(chips_dir / 'manifest_conditions.csv', index=False)

    cols = [
        'sun_elevation',
        'view_angle',
        'light_haze_percent',
        'heavy_haze_frac',
        'nir_std',
        'bright_px_frac',
    ]
    print(out.groupby('label')[cols].describe(percentiles=[0.1, 0.5, 0.9]).T.round(3).to_string())


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))

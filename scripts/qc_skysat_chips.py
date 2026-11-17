"""Quality checks for the generated SkySat chips: duplicates, empty positives, busy water.

1. Exact duplicates: identical pixel content (hash).
2. Spatial duplicates: chips whose footprints overlap, and the same annotated object (points
   within --same-object-m on the same date) appearing in more than one chip. Planet images an
   area several times seconds apart, so separate captures of one date can repeat a scene.
3. Point contrast: how far each whale/boat point stands out from its surrounding water
   (robust z-score of a local brightness anomaly). Low = nothing distinguishable at the point.
4. Water anomaly: the strongest local anomaly in each water chip. High = maybe a missed object
   (or a whitecap / glint); worth a look.

Writes <chips>/qc/{chips.csv, points.csv, overlaps.csv, same_object.csv, set_aside_plan.csv}.
With --apply, the planned chips (duplicates, water to review) move to <chips>/set_aside/.

Run: uv run --group data python \
       scripts/qc_skysat_chips.py data/skysat_chips_500 --original data/whale/data/images
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage


def anomaly_z(img):
    """Robust z-score map of local brightness anomalies (bright or dark), smoothed over ~2 m."""
    a = img.astype(np.float32)
    valid = ~(a == 0).all(0)
    b = a.mean(0)
    b[~valid] = np.median(b[valid]) if valid.any() else 0
    resid = ndimage.uniform_filter(b, 5) - ndimage.gaussian_filter(b, 15)
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826 + 1e-6
    z = np.abs(resid) / mad
    z[~valid] = 0
    return z


def plan_set_aside(man, pts, same_object_m, water_z_min, water_max_blobs):
    """Chips to move out of the training pool, with a reason for each.

    - duplicate: per date, chips are kept greedily (positives first, then more objects, higher
      certainty, main before uncertain). A chip is dropped if it overlaps a kept chip of the same
      date by more than half, or if every object in it is already in a kept chip.
    - water_review: a water chip whose strongest anomaly is whale-strength and stands nearly
      alone (whitecaps make many anomalies; a missed boat or whale makes one).
    """
    cert = pd.to_numeric(pts.certainty, errors='coerce').fillna(0)
    agg = pts.assign(c=cert).groupby('chip_id').agg(n=('cls', 'size'), cmax=('c', 'max'))
    man = man.merge(agg, left_on='chip_id', right_index=True, how='left').fillna(
        {'n': 0, 'cmax': 0}
    )
    man['priority'] = (man.label != 'water') * 100 + man.n * 2 + man.cmax + (man.subset == 'main')
    pts_by_chip = {k: g for k, g in pts.groupby('chip_id')}

    out = []
    for _, g in man.groupby('date'):
        kept_fp, kept_pts = [], np.zeros((0, 3))
        for r in g.sort_values('priority', ascending=False).itertuples():
            a = (r.xmin, r.ymin, r.xmax, r.ymax)
            area = (a[2] - a[0]) * (a[3] - a[1])
            reason = None
            for b in kept_fp:
                iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
                ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
                if iw * ih / area > 0.5:
                    reason = 'duplicate: overlaps a kept chip by >50%'
                    break
            mine = pts_by_chip.get(r.chip_id)
            if reason is None and mine is not None and len(kept_pts):
                is_whale = (mine.cls == 'whale').to_numpy(float)
                seen = [
                    (
                        (np.hypot(kept_pts[:, 0] - x, kept_pts[:, 1] - y) < same_object_m)
                        & (kept_pts[:, 2] == w)
                    ).any()
                    for x, y, w in zip(mine.wx, mine.wy, is_whale)
                ]
                if all(seen):
                    reason = 'duplicate: every object is in a kept chip'
            if reason is None and r.label == 'water':
                if r.water_max_z >= water_z_min and r.water_blobs <= water_max_blobs:
                    reason = 'water_review: isolated whale-strength anomaly'
            if reason:
                out.append({'chip_id': r.chip_id, 'subset': r.subset, 'reason': reason})
                if reason.startswith('water_review'):
                    kept_fp.append(a)  # still occupies its footprint
                continue
            kept_fp.append(a)
            if mine is not None:
                new = np.c_[mine.wx, mine.wy, (mine.cls == 'whale').to_numpy(float)]
                kept_pts = np.vstack([kept_pts, new])
    return pd.DataFrame(out, columns=['chip_id', 'subset', 'reason'])


def apply_set_aside(chips_dir, plan):
    """Move planned chips to <chips>/set_aside/ and drop them from each subset's CSVs."""
    dest = chips_dir / 'set_aside'
    (dest / 'images').mkdir(parents=True, exist_ok=True)
    moved_man, moved_pts = [], []
    for sub, g in plan.groupby('subset'):
        d = chips_dir / sub
        ids = set(g.chip_id)
        for f in ['manifest.csv', 'points.csv', 'manifest_conditions.csv']:
            if not (d / f).exists():
                continue
            df = pd.read_csv(d / f, dtype=str, keep_default_na=False)
            out = df[df.chip_id.isin(ids)].assign(from_subset=sub)
            if f == 'manifest.csv':
                moved_man.append(out.merge(g[['chip_id', 'reason']], on='chip_id'))
            elif f == 'points.csv':
                moved_pts.append(out)
            df[~df.chip_id.isin(ids)].to_csv(d / f, index=False)
        for cid in ids:
            src = d / 'images' / f'{cid}.tif'
            if src.exists():
                src.rename(dest / 'images' / src.name)
    for name, parts in [('manifest.csv', moved_man), ('points.csv', moved_pts)]:
        new = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if (dest / name).exists():
            new = pd.concat([pd.read_csv(dest / name, dtype=str, keep_default_na=False), new])
        new.to_csv(dest / name, index=False)
    print(f'moved {len(plan)} chips to {dest}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('chips', type=Path, help='folder holding main/ and uncertain_whales/')
    ap.add_argument('--original', type=Path, help='original chip images, for overlap checks')
    ap.add_argument('--same-object-m', type=float, default=15.0)
    ap.add_argument('--point-radius-px', type=int, default=10)
    ap.add_argument('--water-review-z', type=float, default=60.0)
    ap.add_argument('--water-review-max-blobs', type=int, default=2)
    ap.add_argument('--apply', action='store_true', help='move set-aside chips to set_aside/')
    args = ap.parse_args()

    man, pts = [], []
    for sub in ['main', 'uncertain_whales']:
        d = args.chips / sub
        man.append(pd.read_csv(d / 'manifest.csv').assign(subset=sub))
        pts.append(pd.read_csv(d / 'points.csv', dtype={'certainty': str}).assign(subset=sub))
    man = pd.concat(man, ignore_index=True)
    pts = pd.concat(pts, ignore_index=True)
    pts = pts[pts.cls.isin(['whale', 'boat'])].copy()

    out = args.chips / 'qc'
    out.mkdir(exist_ok=True)

    # ---- per-chip pass: hash, water anomaly, point contrast
    hashes, water_z, water_xy, water_blobs, bg_z = [], [], [], [], []
    point_z = np.full(len(pts), np.nan)
    pts_by_chip = pts.groupby('chip_id').groups
    r = args.point_radius_px
    rng = np.random.default_rng(0)
    for row in man.itertuples():
        with rasterio.open(args.chips / row.subset / 'images' / f'{row.chip_id}.tif') as s:
            img = s.read()
        hashes.append(hashlib.sha1(img.tobytes()).hexdigest())
        z = anomaly_z(img)
        if row.label == 'water':
            k = np.argmax(z)
            water_z.append(float(z.flat[k]))
            water_xy.append(divmod(int(k), z.shape[1])[::-1])
            water_blobs.append(ndimage.label(z > 30)[1])
            # Background reference: the same point statistic at random spots in water chips.
            for _ in range(5):
                x, y = rng.integers(r, z.shape[1] - r, 2)
                bg_z.append(float(z[y - r : y + r + 1, x - r : x + r + 1].max()))
        else:
            water_z.append(np.nan)
            water_xy.append((np.nan, np.nan))
            water_blobs.append(np.nan)
        for i in pts_by_chip.get(row.chip_id, []):
            x, y = int(round(pts.at[i, 'px'])), int(round(pts.at[i, 'py']))
            win = z[max(y - r, 0) : y + r + 1, max(x - r, 0) : x + r + 1]
            point_z[pts.index.get_loc(i)] = float(win.max()) if win.size else np.nan

    man['sha1'] = hashes
    man['water_max_z'] = water_z
    man['water_max_px'] = [xy[0] for xy in water_xy]
    man['water_max_py'] = [xy[1] for xy in water_xy]
    man['water_blobs'] = water_blobs
    pts['point_z'] = point_z
    bg = np.array(bg_z)
    pts['point_z_pct_of_bg'] = [float((bg < v).mean()) if v == v else np.nan for v in point_z]
    print(
        f'background (random water spot) z: median={np.median(bg):.1f}  '
        f'p95={np.percentile(bg, 95):.1f}'
    )

    # ---- exact duplicates
    dup = man[man.duplicated('sha1', keep=False)]
    print(f'exact duplicate chips: {len(dup)}')

    # ---- footprint overlaps (new vs new, new vs original)
    fp = man[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
    ov = []
    for i in range(len(fp)):
        a = fp[i]
        j = np.arange(i + 1, len(fp))
        b = fp[j]
        iw = np.clip(np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0]), 0, None)
        ih = np.clip(np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1]), 0, None)
        inter = iw * ih
        for jj in j[inter > 0]:
            ov.append((i, jj, inter[jj - i - 1] / (250 * 250)))
    ov = pd.DataFrame(ov, columns=['i', 'j', 'frac'])
    if len(ov):
        for c in ['chip_id', 'label', 'capture', 'date', 'subset']:
            ov[f'a_{c}'] = man[c].to_numpy()[ov.i]
            ov[f'b_{c}'] = man[c].to_numpy()[ov.j]
        ov['same_date'] = ov.a_date == ov.b_date
        ov['same_capture'] = ov.a_capture == ov.b_capture
    ov.drop(columns=['i', 'j']).to_csv(out / 'overlaps.csv', index=False)
    print(f'overlapping new-chip pairs: {len(ov)}')
    if len(ov):
        print(ov.groupby(['same_date', 'same_capture']).frac.describe()[['count', 'mean', 'max']])

    if args.original:
        orig = []
        for f in sorted(args.original.glob('*.tif')):
            with rasterio.open(f) as s:
                orig.append(tuple(s.bounds))
        orig = np.array(orig)
        hit = (
            (fp[:, None, 0] < orig[None, :, 2])
            & (fp[:, None, 2] > orig[None, :, 0])
            & (fp[:, None, 1] < orig[None, :, 3])
            & (fp[:, None, 3] > orig[None, :, 1])
        ).any(1)
        man['overlaps_original'] = hit
        print(f'new chips overlapping an original chip: {hit.sum()}')

    # ---- same annotated object in several chips (world coords, same date)
    m = man.set_index('chip_id')
    pts['wx'] = m.loc[pts.chip_id, 'xmin'].to_numpy() + pts.px * 0.5
    pts['wy'] = m.loc[pts.chip_id, 'ymax'].to_numpy() - pts.py * 0.5
    pts['date'] = m.loc[pts.chip_id, 'date'].to_numpy()
    pts['capture'] = m.loc[pts.chip_id, 'capture'].to_numpy()
    same = []
    for (date, cls), g in pts.groupby(['date', 'cls']):
        xy = g[['wx', 'wy']].to_numpy()
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        ii, jj = np.nonzero(np.triu(d < args.same_object_m, 1))
        for a, b in zip(ii, jj):
            ga, gb = g.iloc[a], g.iloc[b]
            if ga.chip_id != gb.chip_id:
                same.append(
                    {
                        'date': date,
                        'cls': cls,
                        'dist_m': round(d[a, b], 1),
                        'a_chip': ga.chip_id,
                        'b_chip': gb.chip_id,
                        'a_capture': ga.capture,
                        'b_capture': gb.capture,
                    }
                )
    same = pd.DataFrame(same)
    same.to_csv(out / 'same_object.csv', index=False)
    print(f'same object in 2+ chips: {len(same)} pairs')
    if len(same):
        print(same.groupby(['cls', same.a_capture == same.b_capture]).size())

    man.to_csv(out / 'chips.csv', index=False)
    pts.to_csv(out / 'points.csv', index=False)

    print('\npoint contrast (z) by class and certainty:')
    print(
        pts.groupby(['cls', 'certainty'], dropna=False)
        .point_z.describe(percentiles=[0.1, 0.5])[['count', '10%', '50%', 'mean']]
        .round(1)
    )
    w = man[man.label == 'water'].water_max_z
    print(f'\nwater max z: median={w.median():.1f} p90={w.quantile(0.9):.1f} max={w.max():.1f}')

    plan = plan_set_aside(
        man, pts, args.same_object_m, args.water_review_z, args.water_review_max_blobs
    )
    plan.to_csv(out / 'set_aside_plan.csv', index=False)
    print(f'\nset-aside plan: {len(plan)} chips')
    if len(plan):
        print(plan.groupby(['subset', 'reason']).size().to_string())
    if args.apply and len(plan):
        apply_set_aside(args.chips, plan)


if __name__ == '__main__':
    main()

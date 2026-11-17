"""Cut 512x512 SkySat chips (verified water, whale, boat) from the annotated source scenes.

Source data (read only, never modified):
  - Scenes:      <drive>/SatWhale/data/usa-hawaii_west_lahaina_c/<date>/**/<scene>.tif
  - Annotations: local copy of <drive>/SatWhale/WhaleDetect/<scene>/<scene>/
                 {Whales,Boats,POI,grid}.shp

How chips are chosen:
  - water: windows fully inside grid cells the annotator marked Done, with no Whale/Boat/POI
    point within --buffer-m, mostly water (NDWI), clear sky (UDM2), and no nodata.
  - whale / boat: one window per annotated point, randomly offset so the object is not always
    centred. Points already inside a chip from the same capture are not chipped again.
  - Every new chip is kept off the footprints of the existing chips in --exclude-chips, so it
    can't duplicate (or leak into) the current dataset.

Only the pixels of each chosen window are read from the drive. Outputs go to --out:
  images/<chip_id>.tif   4-band uint16, georeferenced, LZW
  manifest.csv           one row per chip (label, scene, capture, quality stats)
  points.csv             every Whale/Boat/POI point inside each chip, in chip pixel coords

Run with:  uv run --group data python scripts/build_skysat_chips.py ...
"""

import argparse
import csv
import os
import re
from pathlib import Path

os.environ.setdefault('GDAL_PAM_ENABLED', 'NO')  # don't let GDAL write .aux.xml next to sources

import numpy as np
import rasterio
import shapefile  # pyshp
from rasterio.windows import Window

CHIP = 512
POINT_LAYERS = ('Whales', 'Boats', 'POI')
LABEL_OF = {'Whales': 'whale', 'Boats': 'boat', 'POI': 'poi'}


def read_points(scene_dir):
    """All annotated points in a scene as a list of dicts (x, y in scene CRS)."""
    pts = []
    for layer in POINT_LAYERS:
        path = scene_dir / f'{layer}.shp'
        if not path.exists():
            continue
        r = shapefile.Reader(str(path))
        for i in range(len(r)):
            try:
                shp = r.shape(i)
            except Exception:  # deleted / empty geometry
                continue
            if not shp.points:
                continue
            rec = r.record(i).as_dict()
            x, y = shp.points[0][:2]
            pts.append(
                {
                    'cls': LABEL_OF[layer],
                    'x': x,
                    'y': y,
                    'fid': i,
                    'certainty': str(rec.get('certainty', '') or '').strip(),
                    'taxa': str(rec.get('taxa', '') or '').strip(),
                    'comment': str(rec.get('comment', '') or '').strip(),
                }
            )
    return pts


def read_done_cells(scene_dir):
    """Bounding boxes (xmin, ymin, xmax, ymax) of grid cells marked Done ('y...')."""
    path = scene_dir / 'grid.shp'
    if not path.exists():
        return np.zeros((0, 4))
    r = shapefile.Reader(str(path))
    if 'Done' not in [f[0] for f in r.fields[1:]]:
        return np.zeros((0, 4))
    boxes = []
    for i in range(len(r)):
        done = str(r.record(i)['Done'] or '').strip().lower()
        if not done.startswith('y'):
            continue
        try:
            boxes.append(r.shape(i).bbox)
        except Exception:
            continue
    return np.array(boxes, dtype=float).reshape(-1, 4)


def inside_any(xy, boxes):
    """For each (x, y) in xy, whether it falls in any box."""
    if len(boxes) == 0:
        return np.zeros(len(xy), bool)
    x, y = xy[:, :1], xy[:, 1:]
    return ((x >= boxes[:, 0]) & (x <= boxes[:, 2]) & (y >= boxes[:, 1]) & (y <= boxes[:, 3])).any(
        1
    )


def intersects_any(fp, boxes):
    """For each footprint (xmin, ymin, xmax, ymax), whether it overlaps any box."""
    if len(boxes) == 0:
        return np.zeros(len(fp), bool)
    return (
        (fp[:, None, 0] < boxes[None, :, 2])
        & (fp[:, None, 2] > boxes[None, :, 0])
        & (fp[:, None, 1] < boxes[None, :, 3])
        & (fp[:, None, 3] > boxes[None, :, 1])
    ).any(1)


def footprint(transform, col, row, size=CHIP):
    x0, y0 = transform * (col, row)
    x1, y1 = transform * (col + size, row + size)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def find_sources(scenes_root):
    tifs = {}
    for p in scenes_root.rglob('*.tif'):
        tifs.setdefault(p.name, p)
    return tifs


def udm2_for(src_path):
    name = src_path.name
    cands = [
        name.replace('_pansharpened.tif', '_pansharpened_udm2.tif'),
        name.replace('_pansharpened_clip.tif', '_pansharpened_udm2_clip.tif'),
    ]
    for c in cands:
        p = src_path.with_name(c)
        if c != name and p.exists():
            return p
    return None


def capture_of(scene):
    """Scene name without the u000N segment: overlapping segments of one capture share it."""
    return re.sub(r'_u\d+_.*$', '', scene)


def chip_stats(data, udm_src, src_transform, col, row):
    valid = ~(data == 0).all(0)
    g = data[1].astype(np.float32)
    nir = data[3].astype(np.float32)
    ndwi = (g - nir) / (g + nir + 1e-6)
    stats = {
        'nodata_frac': round(float(1 - valid.mean()), 4),
        'water_frac': round(float(((ndwi > 0) & valid).mean()), 4),
        'clear_frac': '',
        'cloud_frac': '',
        'heavy_haze_frac': '',
    }
    if udm_src is not None:
        # UDM2 bands: 1 clear, 2 snow, 3 shadow, 4 light haze, 5 heavy haze, 6 cloud.
        # Only cloud/shadow/snow reject a water chip. Haze is just recorded: whole 2023 scenes are
        # flagged light or heavy haze yet were annotated (64 whales in one), so dropping hazy water
        # would leave haze present only in positive chips, a label shortcut. Filter on
        # heavy_haze_frac later if wanted, but do it for every class alike.
        fp = footprint(src_transform, col, row)
        win = (
            rasterio.windows.from_bounds(*fp, transform=udm_src.transform)
            .round_offsets()
            .round_lengths()
        )
        u = udm_src.read([1, 2, 3, 5, 6], window=win, boundless=True, fill_value=0)
        if u.size:
            stats['clear_frac'] = round(float((u[0] == 1).mean()), 4)
            stats['cloud_frac'] = round(float((u[[1, 2, 4]] == 1).any(0).mean()), 4)
            stats['heavy_haze_frac'] = round(float((u[3] == 1).mean()), 4)
    return stats


def coarse_water_map(src, factor=16):
    """Low-res (from overviews) valid & water masks for the whole scene, for cheap prefiltering."""
    h, w = src.height // factor, src.width // factor
    g, nir = src.read([2, 4], out_shape=(2, h, w)).astype(np.float32)
    valid = (g > 0) & (nir > 0)
    water = valid & ((g - nir) / (g + nir + 1e-6) > 0)
    return valid, water


def write_chip(path, data, src, col, row):
    profile = {
        'driver': 'GTiff',
        'width': CHIP,
        'height': CHIP,
        'count': src.count,
        'dtype': src.dtypes[0],
        'crs': src.crs,
        'transform': src.window_transform(Window(col, row, CHIP, CHIP)),
        'nodata': src.nodata,
        'compress': 'lzw',
        'predictor': 2,
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
    }
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)
        dst.descriptions = src.descriptions


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument('--scenes-root', type=Path, required=True)
    ap.add_argument('--annotations', type=Path, required=True)
    ap.add_argument(
        '--exclude-chips', type=Path, help='folder of existing chip .tifs to stay clear of'
    )
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mode', choices=['plan', 'extract'], default='plan')
    ap.add_argument('--water-per-scene', type=int, default=20)
    ap.add_argument(
        '--pos-per-scene', type=int, default=10**6, help='cap on whale/boat chips per scene'
    )
    ap.add_argument('--scenes', nargs='*', help='only these scene names')
    ap.add_argument(
        '--buffer-m', type=float, default=50.0, help='water chips keep this far from any point'
    )
    ap.add_argument('--min-water-frac', type=float, default=0.98)
    ap.add_argument(
        '--max-cloud-frac',
        type=float,
        default=0.02,
        help='UDM2 cloud/shadow/snow, water chips only',
    )
    ap.add_argument('--max-pos-nodata', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sources = find_sources(args.scenes_root)

    exclude = []
    if args.exclude_chips:
        for f in sorted(args.exclude_chips.glob('*.tif')):
            with rasterio.open(f) as s:
                exclude.append(tuple(s.bounds))
    exclude = np.array(exclude, dtype=float).reshape(-1, 4)

    if args.mode == 'extract':
        (args.out / 'images').mkdir(parents=True, exist_ok=True)
        man_f = open(args.out / 'manifest.csv', 'w', newline='')
        pts_f = open(args.out / 'points.csv', 'w', newline='')
        man = csv.DictWriter(
            man_f,
            fieldnames=[
                'chip_id',
                'label',
                'n_whale',
                'n_boat',
                'n_poi',
                'whale_certainty_max',
                'scene',
                'capture',
                'date',
                'col_off',
                'row_off',
                'xmin',
                'ymin',
                'xmax',
                'ymax',
                'done_frac',
                'nodata_frac',
                'water_frac',
                'clear_frac',
                'cloud_frac',
                'heavy_haze_frac',
            ],
        )
        man.writeheader()
        pts_w = csv.DictWriter(
            pts_f,
            fieldnames=['chip_id', 'cls', 'px', 'py', 'certainty', 'taxa', 'scene_fid', 'comment'],
        )
        pts_w.writeheader()

    taken_by_capture = {}  # capture -> list of footprints already chipped
    totals = {'water_cand': 0, 'pos_points': 0, 'water': 0, 'whale': 0, 'boat': 0, 'rejected': 0}
    plan_rows = []

    scene_dirs = sorted(p for p in args.annotations.iterdir() if p.is_dir())
    if args.scenes:
        scene_dirs = [p for p in scene_dirs if p.name in set(args.scenes)]

    for sdir in scene_dirs:
        scene = sdir.name
        src_path = sources.get(scene + '.tif')
        if src_path is None:
            continue
        with rasterio.open(src_path) as src:
            if src.dtypes[0] != 'uint16' or src.count != 4 or 'analytic' in scene:
                continue  # visual (uint8) / surface reflectance: wrong format
            udm_path = udm2_for(src_path)
            udm_src = rasterio.open(udm_path) if udm_path else None
            capture = capture_of(scene)
            date = scene[:8]
            taken = taken_by_capture.setdefault(capture, [])
            T = src.transform
            pts = read_points(sdir)
            pxy = np.array([[p['x'], p['y']] for p in pts], dtype=float).reshape(-1, 2)
            done = read_done_cells(sdir)

            def points_in(fp):
                m = (
                    (pxy[:, 0] >= fp[0])
                    & (pxy[:, 0] < fp[2])
                    & (pxy[:, 1] > fp[1])
                    & (pxy[:, 1] <= fp[3])
                )
                return [pts[i] for i in np.flatnonzero(m)]

            def done_frac(fp):
                xs = np.linspace(fp[0], fp[2], 9)
                ys = np.linspace(fp[1], fp[3], 9)
                g = np.array([[x, y] for x in xs for y in ys])
                return round(float(inside_any(g, done).mean()), 3)

            def emit(label, col, row, data, fp, stats, inside):
                chip_id = f'{scene}_c{col}_r{row}'
                write_chip(args.out / 'images' / f'{chip_id}.tif', data, src, col, row)
                n = {c: sum(p['cls'] == c for p in inside) for c in ('whale', 'boat', 'poi')}
                certs = [
                    int(p['certainty'])
                    for p in inside
                    if p['cls'] == 'whale' and p['certainty'].isdigit()
                ]
                man.writerow(
                    {
                        'chip_id': chip_id,
                        'label': label,
                        'n_whale': n['whale'],
                        'n_boat': n['boat'],
                        'n_poi': n['poi'],
                        'whale_certainty_max': max(certs) if certs else '',
                        'scene': scene,
                        'capture': capture,
                        'date': date,
                        'col_off': col,
                        'row_off': row,
                        'xmin': fp[0],
                        'ymin': fp[1],
                        'xmax': fp[2],
                        'ymax': fp[3],
                        'done_frac': done_frac(fp),
                        **stats,
                    }
                )
                inv = ~T
                for p in inside:
                    c, r = inv * (p['x'], p['y'])
                    pts_w.writerow(
                        {
                            'chip_id': chip_id,
                            'cls': p['cls'],
                            'px': round(c - col, 1),
                            'py': round(r - row, 1),
                            'certainty': p['certainty'],
                            'taxa': p['taxa'],
                            'scene_fid': p['fid'],
                            'comment': p['comment'],
                        }
                    )
                taken.append(fp)

            # ---------------- positives: whale / boat points ----------------
            pos = [p for p in pts if p['cls'] in ('whale', 'boat')]
            rng.shuffle(pos)
            n_pos = 0
            for p in pos:
                if n_pos >= args.pos_per_scene:
                    break
                margin = 32 * abs(T.a)
                if any(
                    fp[0] + margin <= p['x'] <= fp[2] - margin
                    and fp[1] + margin <= p['y'] <= fp[3] - margin
                    for fp in taken
                ):
                    continue  # already inside a chip from this capture
                c, r = ~T * (p['x'], p['y'])
                col = int(round(c - CHIP / 2 + rng.integers(-160, 161)))
                row = int(round(r - CHIP / 2 + rng.integers(-160, 161)))
                col = min(max(col, 0), src.width - CHIP)
                row = min(max(row, 0), src.height - CHIP)
                fp = footprint(T, col, row)
                if intersects_any(np.array([fp]), exclude)[0]:
                    totals['rejected'] += 1
                    continue
                totals['pos_points'] += 1
                if args.mode == 'plan':
                    taken.append(fp)
                    continue
                data = src.read(window=Window(col, row, CHIP, CHIP))
                stats = chip_stats(data, udm_src, T, col, row)
                if stats['nodata_frac'] > args.max_pos_nodata:
                    totals['rejected'] += 1
                    continue
                inside = points_in(fp)
                has_w = any(q['cls'] == 'whale' for q in inside)
                has_b = any(q['cls'] == 'boat' for q in inside)
                label = 'whale+boat' if has_w and has_b else ('whale' if has_w else 'boat')
                emit(label, col, row, data, fp, stats, inside)
                totals['whale' if has_w else 'boat'] += 1
                n_pos += 1

            # ---------------- water: verified-empty windows in Done cells ----------------
            n_water_cand = 0
            if len(done):
                cols = np.arange(0, src.width - CHIP + 1, CHIP)
                rows = np.arange(0, src.height - CHIP + 1, CHIP)
                cc, rr = np.meshgrid(cols, rows)
                cc, rr = cc.ravel(), rr.ravel()
                x0, y0 = T * (cc, rr)
                x1, y1 = T * (cc + CHIP, rr + CHIP)
                fps = np.stack(
                    [
                        np.minimum(x0, x1),
                        np.minimum(y0, y1),
                        np.maximum(x0, x1),
                        np.maximum(y0, y1),
                    ],
                    1,
                )
                ok = np.ones(len(fps), bool)
                for cx, cy in [(0, 1), (2, 1), (0, 3), (2, 3)]:  # every corner inside a Done cell
                    ok &= inside_any(fps[:, [cx, cy]], done)
                if len(pxy):  # no annotated point within the buffer
                    b = args.buffer_m
                    near = (
                        (pxy[None, :, 0] >= fps[:, None, 0] - b)
                        & (pxy[None, :, 0] <= fps[:, None, 2] + b)
                        & (pxy[None, :, 1] >= fps[:, None, 1] - b)
                        & (pxy[None, :, 1] <= fps[:, None, 3] + b)
                    ).any(1)
                    ok &= ~near
                ok &= ~intersects_any(fps, exclude)
                if args.mode == 'extract' and ok.any():
                    # Done cells cover nodata collar and land too: prefilter on a low-res read.
                    f = 16
                    valid, water = coarse_water_map(src, f)
                    k = CHIP // f
                    for i in np.flatnonzero(ok):
                        r0, c0 = rr[i] // f, cc[i] // f
                        v = valid[r0 : r0 + k, c0 : c0 + k]
                        if (
                            v.size == 0
                            or not v.all()
                            or water[r0 : r0 + k, c0 : c0 + k].mean() < args.min_water_frac
                        ):
                            ok[i] = False
                if taken:
                    ok &= ~intersects_any(fps, np.array(taken))
                idx = np.flatnonzero(ok)
                n_water_cand = len(idx)
                totals['water_cand'] += n_water_cand
                if args.mode == 'extract':
                    rng.shuffle(idx)
                    n_water, tries = 0, 0
                    for i in idx:
                        if n_water >= args.water_per_scene or tries >= 4 * args.water_per_scene:
                            break
                        tries += 1
                        col, row, fp = int(cc[i]), int(rr[i]), tuple(fps[i])
                        data = src.read(window=Window(col, row, CHIP, CHIP))
                        stats = chip_stats(data, udm_src, T, col, row)
                        sky_ok = (
                            stats['cloud_frac'] == '' or stats['cloud_frac'] <= args.max_cloud_frac
                        )
                        if (
                            stats['nodata_frac'] > 0
                            or stats['water_frac'] < args.min_water_frac
                            or not sky_ok
                        ):
                            totals['rejected'] += 1
                            continue
                        emit('water', col, row, data, fp, stats, [])
                        n_water += 1
                        totals['water'] += 1

            plan_rows.append(
                (
                    scene,
                    len(done),
                    sum(p['cls'] == 'whale' for p in pts),
                    sum(p['cls'] == 'boat' for p in pts),
                    n_water_cand,
                )
            )
            if udm_src is not None:
                udm_src.close()
        print(
            f'{scene}: done_cells={len(done)} water_candidates={n_water_cand} | running {totals}',
            flush=True,
        )

    if args.mode == 'extract':
        man_f.close()
        pts_f.close()
    print('TOTALS', totals)


if __name__ == '__main__':
    main()

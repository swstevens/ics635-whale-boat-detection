"""Shared evaluation: COCO mAP plus scene-level bootstrap confidence intervals.

Every model writes its test-set predictions in COCO results format,
    [{"image_id": int, "category_id": int, "bbox": [x, y, w, h], "score": float}, ...]
and is scored with this module, so the numbers are directly comparable.

Bootstrap resamples whole scenes (not chips), since chips from one scene are
correlated. When several prediction files are given, every model is scored on
the same resamples, so the CI on the difference between two models is paired.

Usage:
    python -m whaleboat.eval data/splits/test.json results/cnn/preds_test.json \
        results/detr/preds_test.json --bootstrap 200 --out results/comparison.json
"""

import argparse
import contextlib
import io
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

STAT_NAMES = [
    'AP',
    'AP50',
    'AP75',
    'AP_small',
    'AP_medium',
    'AP_large',
    'AR1',
    'AR10',
    'AR100',
    'AR_small',
    'AR_medium',
    'AR_large',
]
# Metrics shown in the printed table; everything is still saved to --out.
SHOWN = ['AP', 'AP50', 'AP_small', 'AR100']


def _normalize_gt(gt):
    """Fill in the fields COCOeval requires but annotation exports often omit."""
    for a in gt['annotations']:
        a.setdefault('area', a['bbox'][2] * a['bbox'][3])
        a.setdefault('iscrowd', 0)
    return gt


def _coco(dataset):
    coco = COCO()
    coco.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def _mean_valid(x):
    x = x[x > -1]
    return float(x.mean()) if x.size else float('nan')


def evaluate(gt, preds):
    """Score one set of predictions against a COCO ground-truth dict."""
    coco_gt = _coco(gt)
    if not preds:
        return {name: 0.0 for name in STAT_NAMES}
    with contextlib.redirect_stdout(io.StringIO()):
        # loadRes edits the dicts it is given, so hand it copies.
        coco_dt = coco_gt.loadRes([dict(p) for p in preds])
        E = COCOeval(coco_gt, coco_dt, 'bbox')
        E.evaluate()
        E.accumulate()
        E.summarize()
    out = dict(zip(STAT_NAMES, E.stats.tolist()))

    # precision has shape [iou_thresholds, recall, category, area_range, max_dets];
    # area_range 0 is "all" and max_dets -1 is 100.
    precision = E.eval['precision']
    for k, cat_id in enumerate(E.params.catIds):
        name = coco_gt.cats[cat_id]['name']
        out[f'AP_{name}'] = _mean_valid(precision[:, :, k, 0, -1])
        out[f'AP50_{name}'] = _mean_valid(precision[0, :, k, 0, -1])
    return out


def _by_image(items):
    grouped = defaultdict(list)
    for item in items:
        grouped[item['image_id']].append(item)
    return grouped


def bootstrap(gt, pred_sets, n=200, seed=0, group_key='scene_id'):
    """Percentile CIs for each model, and paired CIs for each pairwise difference.

    pred_sets maps a model name to its list of predictions. Pass
    group_key=None to resample individual chips instead of scenes.
    """
    groups = defaultdict(list)
    for img in gt['images']:
        if group_key is not None and group_key not in img:
            raise ValueError(f"Image {img['id']} has no '{group_key}'; see DATA.md.")
        groups[img[group_key] if group_key else img['id']].append(img)
    keys = list(groups)
    anns_by_image = _by_image(gt['annotations'])
    preds_by_image = {name: _by_image(p) for name, p in pred_sets.items()}

    rng = np.random.default_rng(seed)
    samples = {name: defaultdict(list) for name in pred_sets}
    for _ in range(n):
        images, anns = [], []
        preds = {name: [] for name in pred_sets}
        # A scene drawn twice needs fresh ids, or COCO would merge the copies.
        for g in rng.choice(len(keys), size=len(keys), replace=True):
            for img in groups[keys[g]]:
                new_id = len(images) + 1
                images.append({**img, 'id': new_id})
                for a in anns_by_image[img['id']]:
                    anns.append({**a, 'id': len(anns) + 1, 'image_id': new_id})
                for name in pred_sets:
                    preds[name].extend(
                        {**p, 'image_id': new_id} for p in preds_by_image[name][img['id']]
                    )
        sample_gt = {**gt, 'images': images, 'annotations': anns}
        for name in pred_sets:
            for metric, value in evaluate(sample_gt, preds[name]).items():
                samples[name][metric].append(value)

    def ci(values):
        values = np.asarray(values, dtype=float)
        lo, hi = np.nanpercentile(values, [2.5, 97.5])
        return {'mean': float(np.nanmean(values)), 'lo': float(lo), 'hi': float(hi)}

    result = {
        'n_resamples': n,
        'group_key': group_key,
        'ci': {name: {m: ci(v) for m, v in s.items()} for name, s in samples.items()},
        'diff': {},
    }
    for a, b in combinations(pred_sets, 2):
        result['diff'][f'{b} - {a}'] = {
            m: ci(np.subtract(samples[b][m], samples[a][m])) for m in samples[a]
        }
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('gt', type=Path, help='COCO ground truth for the split being scored')
    parser.add_argument('preds', type=Path, nargs='+', help='COCO results json, one per model')
    parser.add_argument(
        '--bootstrap', type=int, default=0, help='number of resamples (0 = point estimates only)'
    )
    parser.add_argument(
        '--group-key', default='scene_id', help="'none' resamples chips instead of scenes"
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()

    gt = _normalize_gt(json.loads(args.gt.read_text()))
    # Name each model after its results folder, e.g. results/detr/preds_test.json -> "detr".
    pred_sets = {p.parent.name or p.stem: json.loads(p.read_text()) for p in args.preds}
    shown = SHOWN + [f'AP50_{c["name"]}' for c in gt['categories']]

    report = {'point': {name: evaluate(gt, preds) for name, preds in pred_sets.items()}}
    if args.bootstrap:
        group_key = None if args.group_key.lower() == 'none' else args.group_key
        report.update(bootstrap(gt, pred_sets, args.bootstrap, args.seed, group_key))

    for name, metrics in report['point'].items():
        print(f'\n{name}')
        for m in shown:
            line = f'  {m:<14}{metrics[m]:.3f}'
            if args.bootstrap:
                c = report['ci'][name][m]
                line += f'   95% CI [{c["lo"]:.3f}, {c["hi"]:.3f}]'
            print(line)
    for pair, metrics in report.get('diff', {}).items():
        print(f'\n{pair}  (paired; positive means the left-hand model is better)')
        for m in shown:
            c = metrics[m]
            print(f'  {m:<14}{c["mean"]:+.3f}   95% CI [{c["lo"]:+.3f}, {c["hi"]:+.3f}]')

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

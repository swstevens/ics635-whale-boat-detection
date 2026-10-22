"""Scene-grouped train/val/test split of a COCO annotation file.

Chips cut from the same SkySat scene share lighting, sea state, and often the
same animals, so they must all land in the same split. Otherwise test scores
are inflated by near-duplicates seen during training.

Usage:
    python -m whaleboat.splits data/annotations/all.json data/splits
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _subset(coco, image_ids):
    image_ids = set(image_ids)
    return {
        **coco,
        "images": [img for img in coco["images"] if img["id"] in image_ids],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in image_ids],
    }


def _class_counts(coco, image_ids):
    image_ids = set(image_ids)
    return Counter(a["category_id"] for a in coco["annotations"] if a["image_id"] in image_ids)


def split(coco, ratios=(0.7, 0.15, 0.15), seed=635, group_key="scene_id", tries=200):
    """Assign whole scenes to train/val/test.

    Whales are rare and clustered in a few scenes, so a single random shuffle
    can leave the test set with almost none. We draw `tries` shuffles and keep
    the one whose per-class instance fractions are closest to `ratios`.
    """
    missing = [img["file_name"] for img in coco["images"] if group_key not in img]
    if missing:
        raise ValueError(
            f"{len(missing)} images have no '{group_key}' field (e.g. {missing[0]}). "
            "See DATA.md: every image needs the scene it was cut from."
        )

    images_by_scene = {}
    for img in coco["images"]:
        images_by_scene.setdefault(img[group_key], []).append(img["id"])
    scenes = sorted(images_by_scene, key=str)
    n = len(scenes)
    n_test = max(1, round(ratios[2] * n))
    n_val = max(1, round(ratios[1] * n))
    if n_test + n_val >= n:
        raise ValueError(f"Only {n} scenes; too few for a three-way split.")

    totals = _class_counts(coco, [img["id"] for img in coco["images"]])
    target = np.array(ratios)
    rng = np.random.default_rng(seed)
    best, best_err = None, np.inf
    for _ in range(tries):
        order = list(rng.permutation(n))
        parts = {
            "test": [scenes[i] for i in order[:n_test]],
            "val": [scenes[i] for i in order[n_test : n_test + n_val]],
            "train": [scenes[i] for i in order[n_test + n_val :]],
        }
        err = 0.0
        for cat, total in totals.items():
            fracs = np.array(
                [
                    _class_counts(coco, [i for s in parts[p] for i in images_by_scene[s]])[cat]
                    / total
                    for p in ("train", "val", "test")
                ]
            )
            err += np.abs(fracs - target).sum()
        if err < best_err:
            best, best_err = parts, err
    return best, images_by_scene


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("annotations", type=Path, help="COCO json covering all chips")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--ratios", type=float, nargs=3, default=(0.7, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=635)
    parser.add_argument("--group-key", default="scene_id")
    args = parser.parse_args()

    coco = json.loads(args.annotations.read_text())
    parts, images_by_scene = split(coco, args.ratios, args.seed, args.group_key)
    cats = {c["id"]: c["name"] for c in coco["categories"]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, scenes in parts.items():
        ids = [i for s in scenes for i in images_by_scene[s]]
        (args.out_dir / f"{name}.json").write_text(json.dumps(_subset(coco, ids)))
        counts = _class_counts(coco, ids)
        per_class = ", ".join(f"{cats[c]}={counts[c]}" for c in sorted(cats))
        print(f"{name:>5}: {len(scenes):3d} scenes, {len(ids):5d} chips, {per_class}")

    # The manifest pins which scenes are in which split, so the test set
    # stays fixed even if the annotation file is regenerated later.
    manifest = {"seed": args.seed, "group_key": args.group_key, **parts}
    (args.out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()

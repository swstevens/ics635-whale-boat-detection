"""Fake SkySat-style chips in the DATA.md format, for testing pipelines before real data arrives.

Draws ocean texture with dark elongated "whales", bright "boats" with wakes,
and glare patches as distractors. It only checks that loading, training,
prediction and evaluation run end to end. Scores on it mean nothing.

Usage:
    python -m whaleboat.synthetic data/synthetic --scenes 12 --chips-per-scene 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CATEGORIES = [{'id': 1, 'name': 'whale'}, {'id': 2, 'name': 'boat'}]


def _ocean(rng, size, base):
    noise = rng.normal(0, 1, (size // 8, size // 8, 3))
    img = Image.fromarray(np.uint8(np.clip(noise * 40 + 128, 0, 255))).resize(
        (size, size), Image.BICUBIC
    )
    texture = (np.asarray(img, dtype=float) - 128) * 0.15
    fine = rng.normal(0, 4, (size, size, 3))
    return np.clip(base + texture + fine, 0, 255)


def _rotated_ellipse(size, cx, cy, length, width, angle):
    yy, xx = np.mgrid[0:size, 0:size]
    c, s = np.cos(angle), np.sin(angle)
    u = (xx - cx) * c + (yy - cy) * s
    v = -(xx - cx) * s + (yy - cy) * c
    return (u / (length / 2)) ** 2 + (v / (width / 2)) ** 2 <= 1


def _bbox(mask):
    ys, xs = np.nonzero(mask)
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
    ]


def make_chip(rng, size=500):
    """Return (uint8 RGB array, list of (category_id, bbox))."""
    # Each chip gets its own sea color, roughly deep blue to teal.
    base = np.array([20, 60, 90]) + rng.normal(0, 8, 3)
    img = _ocean(rng, size, base)
    objects = []

    for _ in range(rng.integers(0, 4)):
        cx, cy = rng.uniform(40, size - 40, 2)
        length, width = rng.uniform(18, 32), rng.uniform(6, 10)
        mask = _rotated_ellipse(size, cx, cy, length, width, rng.uniform(0, np.pi))
        img[mask] = img[mask] * 0.55 + np.array([60, 70, 75]) * 0.45
        objects.append((1, _bbox(mask)))

    for _ in range(rng.integers(0, 3)):
        cx, cy = rng.uniform(40, size - 40, 2)
        angle = rng.uniform(0, 2 * np.pi)
        length = rng.uniform(10, 22)
        # Wake: a faint lighter streak trailing behind the hull.
        wake = _rotated_ellipse(
            size,
            cx - np.cos(angle) * length * 2,
            cy - np.sin(angle) * length * 2,
            length * 3.5,
            length * 0.6,
            angle,
        )
        img[wake] = img[wake] * 0.75 + 255 * 0.25
        hull = _rotated_ellipse(size, cx, cy, length, length * 0.35, angle)
        img[hull] = np.array([235, 235, 230]) + rng.normal(0, 5, 3)
        objects.append((2, _bbox(hull)))

    out = Image.fromarray(np.uint8(np.clip(img, 0, 255)))
    # Glare: bright blurred blobs that are not objects, the main false-positive risk.
    if rng.random() < 0.4:
        glare = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(glare)
        for _ in range(rng.integers(3, 12)):
            x, y = rng.uniform(0, size, 2)
            r = rng.uniform(3, 15)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=int(rng.uniform(80, 200)))
        glare = glare.filter(ImageFilter.GaussianBlur(4))
        out = Image.composite(Image.new('RGB', (size, size), (240, 240, 240)), out, glare)
    return out, objects


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--scenes', type=int, default=12)
    parser.add_argument('--chips-per-scene', type=int, default=10)
    parser.add_argument('--size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    (args.out_dir / 'images').mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for s in range(args.scenes):
        for k in range(args.chips_per_scene):
            chip, objects = make_chip(rng, args.size)
            image_id = len(images) + 1
            file_name = f'scene{s:02d}_chip{k:03d}.png'
            chip.save(args.out_dir / 'images' / file_name)
            images.append(
                {
                    'id': image_id,
                    'file_name': file_name,
                    'width': args.size,
                    'height': args.size,
                    'scene_id': f'scene{s:02d}',
                }
            )
            for category_id, bbox in objects:
                annotations.append(
                    {
                        'id': len(annotations) + 1,
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox': bbox,
                        'area': bbox[2] * bbox[3],
                        'iscrowd': 0,
                    }
                )

    (args.out_dir / 'annotations').mkdir(exist_ok=True)
    coco = {'images': images, 'annotations': annotations, 'categories': CATEGORIES}
    (args.out_dir / 'annotations' / 'all.json').write_text(json.dumps(coco))
    print(f'{len(images)} chips, {len(annotations)} objects -> {args.out_dir}')


if __name__ == '__main__':
    main()

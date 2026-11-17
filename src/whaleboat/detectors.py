"""The two detectors behind one interface, plus the shared training loop and prediction.

Both wrappers take images as (B, 3, S, S) floats in [0, 1] and targets from DetectionDataset
(xyxy boxes in S pixels, labels indexing DET_CLASSES), and return detections in the same format,
so the notebooks never branch on the model type except to build it.

- FasterRCNN: torchvision Faster R-CNN, ResNet-50 FPN v2 backbone, COCO weights (CNN baseline).
- Detr: Hugging Face DETR-family model, default Conditional DETR (ResNet-50, COCO). Deformable
  DETR ('SenseTime/deformable-detr') can be swapped in by checkpoint name.
"""

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn

from .data import IMAGENET_MEAN, IMAGENET_STD
from .detection import DET_CLASSES


class FasterRCNN(nn.Module):
    """min_anchor: smallest anchor side in S pixels (COCO default 32; whales are ~20-40 px at
    scale 1). trainable_layers: ResNet stages fine-tuned (0-5); the rest stay at COCO weights."""

    kind = 'faster_rcnn'

    def __init__(
        self, img_size, min_anchor=32, trainable_layers=3, pretrained=True, detections_per_img=100
    ):
        super().__init__()
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )
        from torchvision.models.detection.anchor_utils import AnchorGenerator
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

        anchors = AnchorGenerator(
            tuple((min_anchor * 2**i,) for i in range(5)), ((0.5, 1.0, 2.0),) * 5
        )
        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None,
            weights_backbone=None,
            trainable_backbone_layers=trainable_layers,
            # The notebook already sized the image; stop torchvision resizing it again.
            min_size=img_size,
            max_size=img_size,
            # Keep weak detections: the operating threshold is chosen later, for recall.
            box_score_thresh=0.001,
            box_detections_per_img=detections_per_img,
        )
        # Same 3 aspect ratios per location as COCO, so the pretrained RPN head still fits.
        self.model.rpn.anchor_generator = anchors
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(DET_CLASSES) + 1)

    def param_groups(self, lr, backbone_lr_mult=1.0):
        params = [p for p in self.parameters() if p.requires_grad]
        return [{'params': params, 'lr': lr}]

    def loss(self, images, targets):
        t = [{'boxes': x['boxes'], 'labels': x['labels'] + 1} for x in targets]  # 0 = background
        return self.model(list(images), t)

    @torch.no_grad()
    def predict(self, images):
        out = self.model(list(images))
        return [
            {'boxes': o['boxes'], 'scores': o['scores'], 'labels': o['labels'] - 1} for o in out
        ]


class Detr(nn.Module):
    """DETR-family detector from the Hugging Face hub.

    Conditional / Deformable DETR score each query with a sigmoid per class (focal loss), so
    there is no 'no-object' weight to tune; recall is controlled by keeping low-scoring queries
    and choosing the threshold afterwards.
    """

    kind = 'detr'

    def __init__(
        self,
        img_size,
        checkpoint='microsoft/conditional-detr-resnet-50',
        pretrained=True,
        top_k=100,
    ):
        super().__init__()
        from transformers import AutoConfig, AutoModelForObjectDetection

        names = {
            'id2label': dict(enumerate(DET_CLASSES)),
            'label2id': {k: i for i, k in enumerate(DET_CLASSES)},
            # Loss on every decoder layer, as in the DETR papers (off by default in HF).
            'auxiliary_loss': True,
        }
        if pretrained:
            # The COCO class head (91 classes) is replaced with a fresh 2-class head.
            self.model = AutoModelForObjectDetection.from_pretrained(
                checkpoint, ignore_mismatched_sizes=True, **names
            )
        else:
            cfg = AutoConfig.from_pretrained(checkpoint, **names)
            cfg.use_pretrained_backbone = False
            self.model = AutoModelForObjectDetection.from_config(cfg)
        # Focal-loss prior: start every query at ~1% object probability, as the DETR / RetinaNet
        # recipes do. With a fresh zero bias each of the 300 queries starts at 50% and the
        # classification loss swamps the box losses for the first epochs.
        prior = -math.log((1 - 0.01) / 0.01)
        for name, m in self.model.named_modules():
            if (
                isinstance(m, nn.Linear)
                and m.out_features == len(DET_CLASSES)
                and ('class_labels_classifier' in name or 'class_embed' in name)
            ):
                nn.init.constant_(m.bias, prior)
        self.img_size = img_size
        self.top_k = top_k
        self.register_buffer('mean', IMAGENET_MEAN.clone(), persistent=False)
        self.register_buffer('std', IMAGENET_STD.clone(), persistent=False)

    def param_groups(self, lr, backbone_lr_mult=0.1):
        bb, rest = [], []
        for n, p in self.named_parameters():
            if p.requires_grad:
                (bb if 'backbone' in n else rest).append(p)
        return [{'params': rest, 'lr': lr}, {'params': bb, 'lr': lr * backbone_lr_mult}]

    def _inputs(self, images):
        return (images - self.mean) / self.std

    def loss(self, images, targets):
        s = self.img_size
        labels = []
        for t in targets:
            b = t['boxes']
            cxcywh = (
                torch.stack(
                    [
                        (b[:, 0] + b[:, 2]) / 2,
                        (b[:, 1] + b[:, 3]) / 2,
                        b[:, 2] - b[:, 0],
                        b[:, 3] - b[:, 1],
                    ],
                    1,
                )
                / s
            )
            labels.append({'class_labels': t['labels'], 'boxes': cxcywh})
        out = self.model(pixel_values=self._inputs(images), labels=labels)
        # out.loss is the weighted sum the DETR recipe trains on; loss_dict holds the unweighted
        # parts plus a cardinality error that is only a diagnostic, so it isn't summed.
        return {'loss': out.loss}

    def raw(self, images, **kw):
        return self.model(pixel_values=self._inputs(images), **kw)

    def decode(self, out):
        prob = out.logits.sigmoid()  # (B, Q, C)
        b, q, c = prob.shape
        scores, idx = prob.flatten(1).topk(min(self.top_k, q * c), dim=1)
        query, label = idx // c, idx % c
        res = []
        for i in range(b):
            cx, cy, w, h = out.pred_boxes[i, query[i]].unbind(-1)
            boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
            res.append(
                {
                    'boxes': boxes * self.img_size,
                    'scores': scores[i],
                    'labels': label[i],
                    'query': query[i],
                }
            )
        return res

    @torch.no_grad()
    def predict(self, images):
        return self.decode(self.raw(images))


def build_detector(kind, img_size, params, pretrained=True, detr_checkpoint=None):
    if kind == 'faster_rcnn':
        return FasterRCNN(
            img_size,
            min_anchor=params.get('min_anchor', 32),
            trainable_layers=params.get('trainable_layers', 3),
            pretrained=pretrained,
        )
    if kind == 'detr':
        kw = {'checkpoint': detr_checkpoint} if detr_checkpoint else {}
        return Detr(img_size, pretrained=pretrained, **kw)
    raise ValueError(f'unknown detector {kind!r}')


# ---------------------------------------------------------------- prediction


@torch.no_grad()
def predict_loader(det, loader, scale, device, amp=False):
    """name -> list of {'cls', 'x', 'y', 'score'} with box centers in chip (500 px) pixels."""
    det.eval()
    preds = {}
    for images, targets in loader:
        with torch.autocast(device.type, enabled=amp and device.type == 'cuda'):
            out = det.predict(images.to(device, non_blocking=True))
        for t, o in zip(targets, out):
            boxes = o['boxes'].float().cpu().numpy() / scale
            cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
            preds[t['name']] = [
                {'cls': DET_CLASSES[int(lab)], 'x': float(x), 'y': float(y), 'score': float(s)}
                for x, y, s, lab in zip(
                    cx, cy, o['scores'].float().cpu().numpy(), o['labels'].cpu().numpy()
                )
                if 0 <= int(lab) < len(DET_CLASSES)
            ]
    return preds


# ---------------------------------------------------------------- training


def fit(
    det,
    train_loader,
    epochs,
    lr,
    weight_decay,
    device,
    evaluate=None,
    backbone_lr_mult=0.1,
    clip=0.1,
    amp=True,
    warmup_frac=0.05,
    patience=None,
    on_epoch=None,
    log=print,
):
    """AdamW with linear warmup then cosine decay; same loop for both detectors.

    evaluate(det) -> dict with 'score' (validation, higher is better) is called after every
    epoch; the best weights are kept and restored at the end. on_epoch(epoch, score) can raise to
    stop (Optuna pruning). Returns the history list.
    """
    det.to(device)
    opt = torch.optim.AdamW(det.param_groups(lr, backbone_lr_mult), weight_decay=weight_decay)
    total = epochs * len(train_loader)
    warm = max(1, int(warmup_frac * total))

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    use_amp = amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    history, best, best_state, since_best = [], -math.inf, None, 0
    for epoch in range(1, epochs + 1):
        det.train()
        t0, sums, n = time.time(), {}, 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = [
                {'boxes': t['boxes'].to(device), 'labels': t['labels'].to(device)} for t in targets
            ]
            with torch.autocast(device.type, enabled=use_amp):
                losses = det.loss(images, targets)
                loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'loss is {loss.item()} at epoch {epoch}')
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(det.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            n += 1
            for k, v in losses.items():
                sums[k] = sums.get(k, 0.0) + v.item()
        row = {
            'epoch': epoch,
            'loss': sum(sums.values()) / max(n, 1),
            'lr': opt.param_groups[0]['lr'],
            'seconds': time.time() - t0,
        }
        row.update({f'loss_{k}': v / max(n, 1) for k, v in sums.items()})
        if evaluate is not None:
            row.update({f'val_{k}': v for k, v in evaluate(det).items()})
            score = row['val_score']
            if score > best:
                best, since_best = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in det.state_dict().items()}
                row['best'] = True
            else:
                since_best += 1
        history.append(row)
        if log:
            log(
                f'epoch {epoch:3d}  loss {row["loss"]:.4f}'
                + (f'  val {row["val_score"]:.4f}' if 'val_score' in row else '')
                + f'  {row["seconds"]:.0f}s'
            )
        if on_epoch is not None:
            on_epoch(epoch, row.get('val_score', -row['loss']))
        if patience and since_best >= patience:
            if log:
                log(f'early stop: no improvement in {patience} epochs')
            break
    if best_state is not None:
        det.load_state_dict(best_state)
    return history


def snapshot(det):
    return copy.deepcopy({k: v.detach().cpu() for k, v in det.state_dict().items()})


def seed_everything(seed):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

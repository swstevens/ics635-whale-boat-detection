"""The three chip classifiers: EfficientNet-B0, DeiT-Tiny (timm) and SatViT-V2 + a linear head."""

import sys
from pathlib import Path

import timm
import torch
import torch.nn as nn

MODELS = {
    'efficientnet_b0': {'timm_name': 'efficientnet_b0', 'img_size': 224, 'batch_size': 16},
    'deit_tiny': {'timm_name': 'deit_tiny_patch16_224', 'img_size': 224, 'batch_size': 16},
    # SatViT-V2 takes 256x256 inputs in 8x8 patches (1024 patches); batch 6 is Joe's setting.
    'satvit': {'img_size': 256, 'batch_size': 6},
}


class SatViTClassifier(nn.Module):
    """Joe's SatViT-V2 encoder (only layers 10-11 train) + mean-pooled linear head."""

    def __init__(self, num_classes, satvit_dir, weights=None, encoder_dim=768):
        super().__init__()
        satvit_dir = Path(satvit_dir)
        # Joe's models.py imports SatViT.SatViT_model, which imports transformer_model directly.
        for p in [satvit_dir / 'SatViT', satvit_dir]:
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from models import build_satvit_encoder  # type: ignore[reportMissingImports]

        self.encoder = build_satvit_encoder(device='cpu')
        if weights is not None:
            self.encoder.load_state_dict(torch.load(weights, map_location='cpu'))
        for name, param in self.encoder.named_parameters():
            param.requires_grad = 'encoder.layers.10' in name or 'encoder.layers.11' in name
        self.head = nn.Sequential(nn.LayerNorm(encoder_dim), nn.Linear(encoder_dim, num_classes))

    def forward(self, x):
        b, c, h, w = x.shape
        p = 8
        if h % p or w % p:
            raise ValueError('Input height and width must be divisible by 8.')
        patches = (
            x.reshape(b, c, h // p, p, w // p, p)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b, -1, c * p * p)
        )
        encodings = self.encoder.encode(images_patches=patches)
        return self.head(encodings.mean(dim=1))


def build_model(name, num_classes, satvit_dir=None, satvit_weights=None, pretrained=True):
    """pretrained=False skips downloading/loading weights (used when loading a checkpoint)."""
    if name == 'satvit':
        weights = satvit_weights if pretrained else None
        return SatViTClassifier(num_classes, satvit_dir, weights)
    return timm.create_model(
        MODELS[name]['timm_name'], pretrained=pretrained, num_classes=num_classes
    )

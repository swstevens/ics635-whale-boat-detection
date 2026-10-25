import os
import sys

import numpy as np
import torch.nn as nn
from einops import rearrange

# Add SatViT to path so it can be loaded
# This is necessary because the original SatViT code was not structured as a proper Python package with __init__.py files, so we have to manually add it to the path.
# We also need to temporarily add the deleted np.float, np.int, and np.bool back into NumPy
# so that the older SatViT source code doesn't crash when it tries to use those deprecated types.
sys.path.append(os.path.abspath('SatViT'))
np.float = np.float64
np.int = np.int_
np.bool = np.bool_
from SatViT.SatViT_model import SatViT


# Custom Decoder
# This is the custom Decoder architecture that will take the 1024x768 output from SatViT and upsample it back to a 256x256 flood map.
class FloodSegmentationHead(nn.Module):
    def __init__(self, encoder_dim=768, num_classes=1):
        """
        Takes the flat 1024 sequence from SatViT and upsamples it
        back to a 256x256 spatial flood map.
        """
        super().__init__()
        # Upsample 32x32 -> 64x64
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(encoder_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Upsample 64x64 -> 128x128
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # Upsample 128x128 -> 256x256
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # Final Projection: Collapse 64 channels down to 1 (Flood or No Flood)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    # The forward function takes the flat sequence output from SatViT,
    # reshapes it back into a 2D grid, and then applies the upsampling blocks
    # to produce the final flood map.
    def forward(self, x):
        # Reshape the 1D sequence back into a 2D spatial grid
        # Input 'x' shape: [Batch, 1024, 768]
        # We un-flatten 1024 back into a 32x32 grid.
        x = rearrange(x, 'b (h w) c -> b c h w', h=32, w=32)

        # Current shape: [Batch, 768, 32, 32]

        # Pass through upsampling blocks
        x = self.up1(x)  # Shape becomes: [Batch, 256, 64, 64]
        x = self.up2(x)  # Shape becomes: [Batch, 128, 128, 128]
        x = self.up3(x)  # Shape becomes: [Batch, 64, 256, 256]

        # Final projection to 1 channel map
        out = self.final_conv(x)  # Shape becomes: [Batch, 1, 256, 256]

        return out


# Define the specific dimensions for SatViT-V2
patch_hw = 8  # pixels per patch (both height and width)
num_channels = 15  # total input Sentinel bands
io_dim_var = int(patch_hw * patch_hw * num_channels)  # 8 * 8 * 15 = 960 features per patch


# This function builds the SatViT encoder with the specified dimensions
# and moves it to the given device (CPU or GPU).
def build_satvit_encoder(device):
    return SatViT(
        io_dim=io_dim_var,
        num_patches=1024,
        encoder_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_dim=512,
        decoder_depth=1,
        decoder_num_heads=8,
    ).to(device)

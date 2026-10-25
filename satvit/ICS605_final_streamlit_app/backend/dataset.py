# dataset.py

# This file defines the custom PyTorch Dataset class for loadingSen1Floods11 data.
# It reads the CSV answer key, builds a search engine to find the .tif files,
# and implements the logic to load, crop, normalize, and return image-mask pairs for training.

import os
import random

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


# Load the official dataset splits using our custom Dataset class
class Sen1Floods11Dataset(Dataset):
    def __init__(self, data_folder, csv_path, crop_size=256):
        self.crop_size = crop_size

        # 1. Load the CSV Answer Key
        df = pd.read_csv(csv_path, names=['image_file', 'label_file'])

        # 2. Build a "Search Engine" dictionary of all .tif files
        print('Indexing data folders... this might take a few seconds...')
        file_index = {}
        for root, dirs, files in os.walk(data_folder):
            for file in files:
                if file.endswith('.tif'):
                    file_index[file] = os.path.join(root, file)

        # 3. Safely map paths using the dictionary
        self.s1_paths = []
        self.s2_paths = []
        self.label_paths = []

        # Iterate through the CSV rows and use the file index to find the correct paths for S1, S2, and labels
        for idx, row in df.iterrows():
            s1_file = os.path.basename(row['image_file'])
            label_file = os.path.basename(row['label_file'])

            # The CSV gives us S1, but we need the S2 file name too!
            s2_file = s1_file.replace('S1Hand', 'S2Hand').replace('S1Weak', 'S2Weak')

            # Use our search engine to find exactly where these files are
            self.s1_paths.append(file_index[s1_file])
            self.s2_paths.append(file_index[s2_file])
            self.label_paths.append(file_index[label_file])

    # The length of the dataset is just how many rows we have in the CSV (or how many S1 files we found)
    def __len__(self):
        return len(self.s1_paths)

    # ==========================================
    # HELPER FUNCTIONS
    # ==========================================

    # Loads a .tif file and returns a numpy array, with different handling for images vs masks
    def _load_tiff(self, path, is_mask=False):
        with rasterio.open(path) as src:
            if is_mask:
                # Masks are 1 channel, loaded as 64-bit integers for the loss function
                mask = src.read(1).astype(np.int64)  # Shape: (512, 512)
                # SANITIZE: Sen1Floods11 uses -1 for 'no data'.
                # We force any negative numbers to become 0 so BCE loss doesn't crash.
                mask[mask < 0] = 0
                return mask
            else:
                # Images are multi-channel, loaded as 32-bit floats for normalization
                img = src.read().astype(np.float32)  # Shape: (15, 512, 512)
                # SANITIZE: Convert any 'Not a Number' pixels from the satellite into 0.0
                img = np.nan_to_num(img, nan=0.0)
                return img

    # Randomly crops a 256x256 patch from the 512x512 image and mask, ensuring they stay aligned
    def _random_crop(self, image_array, mask_array):
        # Crops both arrays simultaneously from 512x512 down to the crop_size
        _, h, w = image_array.shape
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)

        cropped_image = image_array[:, top : top + self.crop_size, left : left + self.crop_size]
        cropped_mask = mask_array[top : top + self.crop_size, left : left + self.crop_size]

        return cropped_image, cropped_mask

    # Applies Z-score normalization independently to each of the 15 bands
    def _normalize_image(self, image_array):
        for i in range(image_array.shape[0]):
            band = image_array[i]
            mean_val = np.mean(band)
            std_val = np.std(band)
            image_array[i] = (band - mean_val) / (std_val + 1e-8)
        return image_array

    # ==========================================
    # MAIN PIPELINE
    # ==========================================

    def __getitem__(self, idx):

        # Get the file paths for S1, S2, and the label mask based on the index
        s1_path = self.s1_paths[idx]
        s2_path = self.s2_paths[idx]
        label_path = self.label_paths[idx]

        # Load S1 (2 channels) and S2 (13 channels)
        s1_array = self._load_tiff(s1_path, is_mask=False)
        s2_array = self._load_tiff(s2_path, is_mask=False)

        # Stack them together into a single 15-channel array!
        image_array = np.concatenate([s1_array, s2_array], axis=0)

        # Load the matching label
        label_array = self._load_tiff(label_path, is_mask=True)

        # Crop down to 256x256
        image_patch, label_patch = self._random_crop(image_array, label_array)

        # Normalize the satellite imagery
        image_patch = self._normalize_image(image_patch)

        # Convert to PyTorch Tensors
        image_tensor = torch.from_numpy(image_patch)
        label_tensor = torch.from_numpy(label_patch)

        return image_tensor, label_tensor

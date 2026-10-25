import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from einops import rearrange


# This class provides a set of static methods to validate the integrity of Sentinel-1 and Sentinel-2 image arrays before they are fed into the model. It checks for minimum size, correct band count, spatial alignment between S1 and S2, and the presence of NaN values.
class ImageValidator:
    @staticmethod
    def is_valid_size(image_array, min_h=256, min_w=256):
        """Checks if the array dimensions meet the minimum requirements."""
        _, h, w = image_array.shape
        return h >= min_h and w >= min_w

    @staticmethod
    def is_valid_band_count(image_array, expected_bands):
        """Checks if the array has the exact number of required spectral bands."""
        channels, _, _ = image_array.shape
        return channels == expected_bands

    @staticmethod
    def validate_band_order(descriptions, expected_order):
        """Checks if the internal metadata tags match the expected band order."""
        # If the tuple is entirely None or empty strings, the metadata was stripped
        if not any(descriptions):
            return 'MISSING_METADATA'

        if descriptions == expected_order:
            return 'VALID'

        return 'INVALID_ORDER'

    @staticmethod
    def can_concatenate(s1_array, s2_array):
        """Validates that S1 and S2 have the exact same spatial dimensions before stacking."""
        return s1_array.shape[1:] == s2_array.shape[1:]

    @staticmethod
    def contains_nans(image_array):
        """Checks if the image array contains any Not-A-Number (NaN) values."""
        return np.isnan(image_array).any()


# This class provides methods to load TIFF files,
# apply necessary preprocessing steps such as cropping and normalization,
# and prepare the data for model inference. It handles both Sentinel-1 and Sentinel-2 images,
# ensuring they are in the correct format and aligned properly.
class ImageFixer:
    @staticmethod
    def fix_band_order(image_array, current_descriptions, expected_order):
        """Reorders the numpy array channels to match the expected order based on metadata."""
        try:
            # Find the index of each expected band inside the scrambled current descriptions
            correct_indices = [current_descriptions.index(band) for band in expected_order]
            print(f' -> Reordering bands from {current_descriptions} to {expected_order}')

            # Slice the array using the corrected index list
            return image_array[correct_indices, :, :]
        except ValueError:
            raise ValueError(
                f'Cannot automatically fix band order. The image is missing a required band.\n'
                f'Expected: {expected_order}\n'
                f'Found: {current_descriptions}'
            )

    @staticmethod
    def fix_size_center_crop(image_array, target_h=256, target_w=256):
        """Applies a center crop to bring the image to exact dimensions."""
        _, current_h, current_w = image_array.shape
        if current_h < target_h or current_w < target_w:
            raise ValueError(f'Array is too small to crop to {target_h}x{target_w}')

        start_y = (current_h - target_h) // 2
        start_x = (current_w - target_w) // 2
        return image_array[:, start_y : start_y + target_h, start_x : start_x + target_w]

    @staticmethod
    def force_spatial_match(s1_array, s2_array):
        """
        If S1 and S2 are different sizes (e.g., off by 1 pixel from GEE exports),
        this crops them both to match the smallest shared dimensions.
        """
        min_h = min(s1_array.shape[1], s2_array.shape[1])
        min_w = min(s1_array.shape[2], s2_array.shape[2])

        # Center crop both to the minimum shared size
        s1_fixed = ImageFixer.fix_size_center_crop(s1_array, min_h, min_w)
        s2_fixed = ImageFixer.fix_size_center_crop(s2_array, min_h, min_w)
        return s1_fixed, s2_fixed

    @staticmethod
    def fix_nans(image_array):
        """Replaces all NaN values with 0.0 to prevent mathematical crashes."""
        return np.nan_to_num(image_array, nan=0.0)


# This class provides a method to prepare Sentinel-1 and Sentinel-2 images for model inference.
# It scales the S2 bands, stacks S1 and S2 together, and applies Z-score normalization independently to each band. The output is a clean, normalized 15-channel array ready for the SatViT model.
class ImagePreprocessor:
    @staticmethod
    def prepare_for_model(s1_img, s2_img):
        """
        Takes aligned, sanitized S1 and S2 arrays, scales S2,
        stacks them, and applies Z-score normalization.
        """
        # 1. Scale S2 to match Sen1Floods11
        s2_img = s2_img / 10000.0

        # 2. Stack them (We assume Validator.can_concatenate passed)
        image_array = np.concatenate([s1_img, s2_img], axis=0)

        # 3. Z-Score Normalization
        print('Applying mathematical normalization...')
        for i in range(image_array.shape[0]):
            band = image_array[i]
            mean_val = np.mean(band)
            std_val = np.std(band)
            # Add 1e-8 to prevent division by zero
            image_array[i] = (band - mean_val) / (std_val + 1e-8)

        return image_array


def process_satellite_inputs(s1_tiff_path, s2_tiff_path, target_size=256):
    """
    Master pipeline function that orchestrates the loading, validation,
    fixing, and mathematical preprocessing of raw satellite imagery.
    """
    # The official Sen1Floods11 Sentinel-2 band sequence
    EXPECTED_S2_ORDER = (
        'B1',
        'B2',
        'B3',
        'B4',
        'B5',
        'B6',
        'B7',
        'B8',
        'B8A',
        'B9',
        'B10',
        'B11',
        'B12',
    )

    print('Pipeline Step 1: Loading raw imagery and metadata...')
    with rasterio.open(s1_tiff_path) as src_s1:
        s1_img = src_s1.read().astype(np.float32)

    with rasterio.open(s2_tiff_path) as src_s2:
        s2_img = src_s2.read().astype(np.float32)
        s2_descriptions = src_s2.descriptions  # Extract the metadata tags here!

    # ==========================================
    # THE MISSING STEP: Check Band Counts!
    # ==========================================
    print('Pipeline Step 2: Validating Band Counts...')
    if not ImageValidator.is_valid_band_count(s1_img, expected_bands=2):
        raise ValueError(
            f'Sentinel-1 image must have exactly 2 bands (VV, VH). Found {s1_img.shape[0]}.'
        )
    if not ImageValidator.is_valid_band_count(s2_img, expected_bands=13):
        raise ValueError(f'Sentinel-2 image must have exactly 13 bands. Found {s2_img.shape[0]}.')

    print('Pipeline Step 3: Validating Sentinel-2 Band Order...')
    order_status = ImageValidator.validate_band_order(s2_descriptions, EXPECTED_S2_ORDER)

    if order_status == 'INVALID_ORDER':
        print(' -> Incorrect band sequence detected. Attempting to auto-correct...')
        s2_img = ImageFixer.fix_band_order(s2_img, s2_descriptions, EXPECTED_S2_ORDER)
    elif order_status == 'MISSING_METADATA':
        print(
            ' -> WARNING: No band metadata found. Proceeding with existing order, but results may be degraded if bands are scrambled.'
        )

    print('Pipeline Step 4: Sanitizing data...')
    if ImageValidator.contains_nans(s1_img):
        s1_img = ImageFixer.fix_nans(s1_img)
    if ImageValidator.contains_nans(s2_img):
        s2_img = ImageFixer.fix_nans(s2_img)

    print('Pipeline Step 5: Enforcing spatial alignment...')
    if not ImageValidator.can_concatenate(s1_img, s2_img):
        print(' -> Mismatch detected. Forcing spatial match...')
        s1_img, s2_img = ImageFixer.force_spatial_match(s1_img, s2_img)

    print('Pipeline Step 6: Validating minimum dimensions...')
    # This is where we use the is_valid_size function!
    if not ImageValidator.is_valid_size(
        s1_img, min_h=target_size, min_w=target_size
    ) or not ImageValidator.is_valid_size(s2_img, min_h=target_size, min_w=target_size):
        raise ValueError(
            f'Images are too small. They must be at least {target_size}x{target_size}.'
        )

    print('Pipeline Step 7: Applying exact target crop...')
    s1_img = ImageFixer.fix_size_center_crop(s1_img, target_h=target_size, target_w=target_size)
    s2_img = ImageFixer.fix_size_center_crop(s2_img, target_h=target_size, target_w=target_size)

    print('Pipeline Step 8: Passing to preprocessor...')
    final_array = ImagePreprocessor.prepare_for_model(s1_img, s2_img)

    print('Pipeline Complete! Array is ready for inference.')
    return final_array


# This function executes inference using a pre-processed 15-channel,
# 256x256 numpy array. It converts the array to a PyTorch tensor, chops it into patches,
# encodes it with the SatViT encoder, decodes it to get raw predictions,
# applies a sigmoid to get probabilities,
# and finally thresholds those probabilities to create a binary flood map.
def predict_new_flood(prepared_image_array, encoder, decoder, device):
    """
    Executes inference using a pre-processed 15-channel, 256x256 numpy array.
    """
    # Convert numpy array to Tensor, add batch dimension, send to GPU/CPU
    image_tensor = torch.from_numpy(prepared_image_array).unsqueeze(0).to(device)

    print('Generating prediction...')
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # Chop into patches and encode
        patches = rearrange(image_tensor, 'b c (h i) (w j) -> b (h w) (c i j)', i=8, j=8)
        encodings = encoder.encode(images_patches=patches)

        # Decode and apply sigmoid for probabilities
        raw_preds = decoder(encodings)
        probabilities = torch.sigmoid(raw_preds)

        # Diagnostic prints
        print(f'Highest confidence of water: {probabilities.max().item() * 100:.2f}%')
        print(f'Average confidence of water: {probabilities.mean().item() * 100:.2f}%')

        # Threshold at 50%
        binary_map = (probabilities > 0.5).squeeze().cpu().numpy()

    print('Success!')
    return binary_map


# # This function visualizes the original Sentinel-2 RGB image alongside the predicted flood map.
# def visualize_flood_prediction(s2_tiff_path, predicted_map):
#     """
#     Visualizes the original optical image alongside the predicted flood map.
#     Applies the same 256x256 center crop used during inference for perfect alignment.
#     """
#     with rasterio.open(s2_tiff_path) as src_s2:
#         s2_img = src_s2.read().astype(np.float32)

#     # Force Center Crop to 256x256 to match the model output
#     _, current_h, current_w = s2_img.shape
#     target_h, target_w = 256, 256
#     start_y = (current_h - target_h) // 2
#     start_x = (current_w - target_w) // 2
#     s2_cropped = s2_img[:, start_y:start_y+target_h, start_x:start_x+target_w]

#     # Extract RGB bands.
#     # Extract True Color RGB: Red (B4 -> Index 3), Green (B3 -> Index 2), Blue (B2 -> Index 1)
#     rgb = s2_cropped[[3, 2, 1], :, :]

#     # Transpose from (Channels, Height, Width) to (Height, Width, Channels) for Matplotlib
#     rgb_plot = np.transpose(rgb, (1, 2, 0))

#     # Normalize RGB for visualization using the 2nd and 98th percentiles to handle glare/shadows
#     p2, p98 = np.percentile(rgb_plot, (2, 98))
#     rgb_plot = np.clip((rgb_plot - p2) / (p98 - p2), 0, 1)

#     # Plotting
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#     axes[0].imshow(rgb_plot)
#     axes[0].set_title("Original Optical (Center Cropped)")
#     axes[0].axis("off")

#     axes[1].imshow(predicted_map, cmap='Blues', vmin=0, vmax=1)
#     axes[1].set_title("Predicted Flood Regions")
#     axes[1].axis("off")

#     plt.tight_layout()
#     plt.show()


# This function visualizes the original Sentinel-2 RGB image alongside the predicted flood map.
def visualize_flood_prediction(s2_tiff_path, predicted_map):
    """
    Visualizes the original optical image alongside the predicted flood map.
    Dynamically searches for RGB bands to ensure correct colors even if the raw file is scrambled.
    """
    with rasterio.open(s2_tiff_path) as src_s2:
        s2_img = src_s2.read().astype(np.float32)
        descriptions = src_s2.descriptions

    # Force Center Crop to 256x256 to match the model output
    _, current_h, current_w = s2_img.shape
    target_h, target_w = 256, 256
    start_y = (current_h - target_h) // 2
    start_x = (current_w - target_w) // 2
    s2_cropped = s2_img[:, start_y : start_y + target_h, start_x : start_x + target_w]

    # Dynamically find True Color RGB: Red (B4), Green (B3), Blue (B2)
    if any(descriptions) and all(b in descriptions for b in ('B4', 'B3', 'B2')):
        red_idx = descriptions.index('B4')
        green_idx = descriptions.index('B3')
        blue_idx = descriptions.index('B2')
        rgb = s2_cropped[[red_idx, green_idx, blue_idx], :, :]
    else:
        # Fallback to Sen1Floods11 default if no metadata exists
        print('Warning: No metadata found for visualization. Assuming default band order.')
        rgb = s2_cropped[[3, 2, 1], :, :]

    # Transpose from (Channels, Height, Width) to (Height, Width, Channels) for Matplotlib
    rgb_plot = np.transpose(rgb, (1, 2, 0))

    # Normalize RGB for visualization using the 2nd and 98th percentiles to handle glare/shadows
    p2, p98 = np.percentile(rgb_plot, (2, 98))
    rgb_plot = np.clip((rgb_plot - p2) / (p98 - p2), 0, 1)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(rgb_plot)
    axes[0].set_title('Original Optical (Center Cropped)')
    axes[0].axis('off')

    axes[1].imshow(predicted_map, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('Predicted Flood Regions')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


# Add this to the bottom of SatViT_inference.py
def save_flood_prediction_image(s2_tiff_path, predicted_map, output_filepath):
    """
    Web-safe visualization: Generates the side-by-side plot and saves it to disk
    instead of displaying a blocking window.
    """
    with rasterio.open(s2_tiff_path) as src_s2:
        s2_img = src_s2.read().astype(np.float32)
        descriptions = src_s2.descriptions

    _, current_h, current_w = s2_img.shape
    target_h, target_w = 256, 256
    start_y = (current_h - target_h) // 2
    start_x = (current_w - target_w) // 2
    s2_cropped = s2_img[:, start_y : start_y + target_h, start_x : start_x + target_w]

    if any(descriptions) and all(b in descriptions for b in ('B4', 'B3', 'B2')):
        red_idx = descriptions.index('B4')
        green_idx = descriptions.index('B3')
        blue_idx = descriptions.index('B2')
        rgb = s2_cropped[[red_idx, green_idx, blue_idx], :, :]
    else:
        rgb = s2_cropped[[3, 2, 1], :, :]

    rgb_plot = np.transpose(rgb, (1, 2, 0))
    p2, p98 = np.percentile(rgb_plot, (2, 98))
    rgb_plot = np.clip((rgb_plot - p2) / (p98 - p2), 0, 1)

    # Use 'Agg' backend to prevent Matplotlib from trying to open a window on the server
    import matplotlib

    matplotlib.use('Agg')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(rgb_plot)
    axes[0].set_title('Original Optical')
    axes[0].axis('off')

    axes[1].imshow(predicted_map, cmap='Blues', vmin=0, vmax=1)
    axes[1].set_title('Predicted Flood Regions')
    axes[1].axis('off')

    plt.tight_layout()

    # Save the figure and cleanly close it to free up server RAM
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    plt.savefig(output_filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

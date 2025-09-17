import numpy as np
import cv2
from typing import List, Tuple, Dict, Union


def normalize_patch(
        patch: np.ndarray,
        upper_percentile: float = 99.5,
        lower_percentile: float = 0.5
) -> np.ndarray:
    """
    Normalize a single image patch using robust percentile range.

    Parameters
    ----------
    patch : np.ndarray
        2D image patch.
    upper_percentile : float, optional
        Upper percentile for normalization range.
    lower_percentile : float, optional
        Lower percentile for normalization range.

    Returns
    -------
    np.ndarray
        Patch normalized to ~[0, 1].
    """
    upper_percentile_val = np.percentile(patch, upper_percentile)
    lower_percentile_val = np.percentile(patch, lower_percentile)
    robust_range = np.abs(upper_percentile_val - lower_percentile_val)
    if upper_percentile_val == lower_percentile_val:
        patch = (patch - patch.min()) / (patch.ptp() + 1e-9)
    else:
        patch = (patch - patch.min()) / (robust_range + 1e-9)
    return patch


def split_into_patches_exhaustive(
        scan: np.ndarray,
        pixel_spacing: Union[List[float], float],
        patch_edge_len: Union[int, float] = 26,
        overlap_param: float = 0.4,
        patch_size: Tuple[int, int] = (224, 224),
        using_resnet: bool = True,
) -> Tuple[List[List[np.ndarray]], List[List[Dict]]]:
    """
    Exhaustively split 3D scan volume into resized, normalized patches for detection.

    Parameters
    ----------
    scan : np.ndarray
        3D scan volume (H, W, D).
    pixel_spacing : Union[List[float], float]
        List [row_spacing, col_spacing] in mm, or scalar.
    patch_edge_len : float, optional
        Patch edge in cm (converted to mm/pixels).
    overlap_param : float, optional
        Overlap fraction between patches.
    patch_size : tuple of int, optional
        Output patch size for resizing (height, width).
    using_resnet : bool, optional
        If True, use robust normalization suited for ResNet training.

    Returns
    -------
    patches : List[List[np.ndarray]]
        List per slice, each is a list of normalized patches.
    transform_info_dicts : List[List[Dict]]
        Patch spatial origin info per slice.
    """
    h, w, d = scan.shape
    
    # Legacy scalar handling
    if isinstance(pixel_spacing, (int, float)):
        pixel_spacing = [float(pixel_spacing), float(pixel_spacing)]
    elif isinstance(pixel_spacing, (list, tuple)):
        if len(pixel_spacing) == 1:
            pixel_spacing = [float(pixel_spacing[0]), float(pixel_spacing[0])]
        elif len(pixel_spacing) == 2:
            pixel_spacing = [float(pixel_spacing[0]), float(pixel_spacing[1])]
        else:
            raise ValueError(f"pixel_spacing must be scalar or 2-list, got {pixel_spacing}")

    # Per-axis patch edge calculation (reflecting legacy usage)
    patch_edge_len_row = patch_edge_len
    patch_edge_len_col = patch_edge_len

    if pixel_spacing[0] != -1 and  pixel_spacing[-1] != -1 :  # If spacing provided (not sentinel value)
        patch_edge_len_row = int(patch_edge_len * 10 / pixel_spacing[0])  # cm to mm, mm to pixels
        patch_edge_len_col = int(patch_edge_len * 10 / pixel_spacing[1])

    # Prevent oversized patches
    max_edge_length = min(h, w) - 1
    if patch_edge_len_row > max_edge_length:
        patch_edge_len_row = max_edge_length
    if patch_edge_len_col > max_edge_length:
        patch_edge_len_col = max_edge_length

    effective_patch_edge_len_row = int(patch_edge_len_row * (1 - overlap_param))
    effective_patch_edge_len_col = int(patch_edge_len_col * (1 - overlap_param))

    num_patches_across = (w // effective_patch_edge_len_col) + 1
    num_patches_down = (h // effective_patch_edge_len_row) + 1

    num_patches = num_patches_down * num_patches_across

    transform_info_dicts = [[None] * num_patches for _ in range(d)]
    patches = [[None] * num_patches for _ in range(d)]

    for slice_idx in range(d):
        for i in range(num_patches_across):

            x1 = i * effective_patch_edge_len_col
            x2 = x1 + patch_edge_len_col
            if x2 >= w:
                x2 = -1
                x1 = w - patch_edge_len_col
            for j in range(num_patches_down):
                y1 = j * effective_patch_edge_len_row
                y2 = y1 + patch_edge_len_row
                if y2 >= h:
                    y2 = -1
                    y1 = h - patch_edge_len_row
                    
                this_patch = np.array(scan[y1:y2, x1:x2, slice_idx])
                resized_patch = cv2.resize(
                    this_patch, patch_size, interpolation=cv2.INTER_CUBIC
                )
                resized_patch = np.clip(resized_patch, this_patch.min(), this_patch.max())

                patch_index = i * num_patches_down + j
                if not using_resnet:
                    patches[slice_idx][patch_index] = 0.5 * (
                                (resized_patch - resized_patch.min()) / (resized_patch.ptp() + 1e-8))
                else:
                    patches[slice_idx][patch_index] = normalize_patch(resized_patch)
                transform_info_dicts[slice_idx][patch_index] = {
                    "x1": x1, "x2": x2, "y1": y1, "y2": y2
                }

    return patches, transform_info_dicts

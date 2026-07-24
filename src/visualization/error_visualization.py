"""
Error Visualization

This module computes and visualizes segmentation errors (false positives and false negatives),
helping identify systematic failure patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


def compute_error_masks(prediction: np.ndarray, ground_truth: np.ndarray) -> dict:
    """
    Compute error masks showing true positives, false positives, and false negatives.

    Args:
        prediction: Binary prediction mask (H, W, D)
        ground_truth: Binary ground truth mask (H, W, D)

    Returns:
        Dictionary with binary masks:
            - true_positive: Correctly predicted foreground
            - false_positive: Predicted but not in ground truth
            - false_negative: Ground truth but not predicted
            - true_negative: Correctly predicted background
    """
    pred_bool = prediction.astype(bool)
    gt_bool = ground_truth.astype(bool)

    true_positive = np.logical_and(pred_bool, gt_bool).astype(np.uint8)
    false_positive = np.logical_and(pred_bool, np.logical_not(gt_bool)).astype(np.uint8)
    false_negative = np.logical_and(np.logical_not(pred_bool), gt_bool).astype(np.uint8)
    true_negative = np.logical_and(np.logical_not(pred_bool), np.logical_not(gt_bool)).astype(np.uint8)

    return {
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'true_negative': true_negative
    }


def compute_error_statistics(error_masks: dict, spacing: tuple = (1.0, 1.0, 1.0)) -> dict:
    """
    Quantify error volumes and connected components.

    Args:
        error_masks: Dictionary from compute_error_masks()
        spacing: Voxel spacing in mm (sx, sy, sz)

    Returns:
        Dictionary with error statistics:
            - fp_volume_ml: False positive volume in mL
            - fn_volume_ml: False negative volume in mL
            - fp_num_components: Number of FP connected components
            - fn_num_components: Number of FN connected components
            - fp_avg_component_size_ml: Average FP component size in mL
            - fn_avg_component_size_ml: Average FN component size in mL
    """
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    voxel_volume_ml = voxel_volume_mm3 / 1000.0

    # False positive statistics
    fp_mask = error_masks['false_positive']
    fp_volume_ml = np.sum(fp_mask) * voxel_volume_ml

    fp_labeled, fp_num_components = ndimage.label(fp_mask)
    if fp_num_components > 0:
        fp_component_sizes = [np.sum(fp_labeled == i) for i in range(1, fp_num_components + 1)]
        fp_avg_component_size_ml = np.mean(fp_component_sizes) * voxel_volume_ml
    else:
        fp_avg_component_size_ml = 0.0

    # False negative statistics
    fn_mask = error_masks['false_negative']
    fn_volume_ml = np.sum(fn_mask) * voxel_volume_ml

    fn_labeled, fn_num_components = ndimage.label(fn_mask)
    if fn_num_components > 0:
        fn_component_sizes = [np.sum(fn_labeled == i) for i in range(1, fn_num_components + 1)]
        fn_avg_component_size_ml = np.mean(fn_component_sizes) * voxel_volume_ml
    else:
        fn_avg_component_size_ml = 0.0

    return {
        'fp_volume_ml': fp_volume_ml,
        'fn_volume_ml': fn_volume_ml,
        'fp_num_components': int(fp_num_components),
        'fn_num_components': int(fn_num_components),
        'fp_avg_component_size_ml': fp_avg_component_size_ml,
        'fn_avg_component_size_ml': fn_avg_component_size_ml,
    }


def create_error_rgb_map(error_masks: dict) -> np.ndarray:
    """
    Create RGB visualization of errors.

    Color scheme:
    - Green: True positives
    - Red: False positives
    - Blue: False negatives
    - Black: True negatives / background

    Args:
        error_masks: Dictionary from compute_error_masks()

    Returns:
        RGB image (H, W, D, 3) with values in [0, 1]
    """
    shape = error_masks['true_positive'].shape
    rgb = np.zeros(shape + (3,), dtype=np.float32)

    # Green channel: True positives
    rgb[..., 1] = error_masks['true_positive'].astype(np.float32)

    # Red channel: False positives
    rgb[..., 0] = error_masks['false_positive'].astype(np.float32)

    # Blue channel: False negatives
    rgb[..., 2] = error_masks['false_negative'].astype(np.float32)

    return rgb


def visualize_error_slice(ct_slice: np.ndarray,
                          error_rgb_slice: np.ndarray,
                          slice_idx: int,
                          view: str = 'axial',
                          output_path: str = None,
                          alpha: float = 0.7):
    """
    Visualize errors as colored overlay on a CT slice.

    Args:
        ct_slice: 2D CT slice
        error_rgb_slice: 2D RGB error map (H, W, 3)
        slice_idx: Slice index for labeling
        view: View plane ('axial', 'sagittal', 'coronal')
        output_path: Path to save figure (optional)
        alpha: Transparency of error overlay
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # CT alone
    axes[0].imshow(ct_slice, cmap='gray', origin='lower')
    axes[0].set_title(f'{view.capitalize()} View - CT\n(Slice {slice_idx})')
    axes[0].axis('off')

    # Error map alone
    axes[1].imshow(error_rgb_slice, origin='lower')
    axes[1].set_title(f'Error Map\n(Green=TP, Red=FP, Blue=FN)')
    axes[1].axis('off')

    # CT + Error overlay
    axes[2].imshow(ct_slice, cmap='gray', origin='lower')
    axes[2].imshow(error_rgb_slice, alpha=alpha, origin='lower')
    axes[2].set_title(f'CT + Error Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_error_mip(ct_volume: np.ndarray,
                        error_rgb: np.ndarray,
                        output_path: str = None,
                        alpha: float = 0.7):
    """
    Visualize errors using Maximum Intensity Projection (MIP).

    Creates 3-view MIP (axial, sagittal, coronal) with error overlay.

    Args:
        ct_volume: 3D CT volume (H, W, D)
        error_rgb: 3D RGB error map (H, W, D, 3)
        output_path: Path to save figure (optional)
        alpha: Transparency of error overlay
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Compute MIPs for CT
    ct_axial = np.max(ct_volume, axis=2)
    ct_sagittal = np.max(ct_volume, axis=1)
    ct_coronal = np.max(ct_volume, axis=0)

    # Compute MIPs for error RGB (take max across depth for each channel)
    err_axial = np.max(error_rgb, axis=2)      # (H, W, D, 3) → (H, W, 3)
    err_sagittal = np.max(error_rgb, axis=1)   # (H, W, D, 3) → (H, D, 3)
    err_coronal = np.max(error_rgb, axis=0)    # (H, W, D, 3) → (W, D, 3)

    # Transpose only spatial dimensions, keep RGB channel last
    err_sagittal_t = np.transpose(err_sagittal, (1, 0, 2))  # (H, D, 3) → (D, H, 3)
    err_coronal_t = np.transpose(err_coronal, (1, 0, 2))    # (W, D, 3) → (D, W, 3)

    # Axial view
    axes[0, 0].imshow(ct_axial, cmap='gray', origin='lower')
    axes[0, 0].set_title('Axial MIP - CT')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(ct_axial, cmap='gray', origin='lower')
    axes[1, 0].imshow(err_axial, alpha=alpha, origin='lower')
    axes[1, 0].set_title('Axial MIP - CT + Errors\n(Green=TP, Red=FP, Blue=FN)')
    axes[1, 0].axis('off')

    # Sagittal view
    axes[0, 1].imshow(ct_sagittal.T, cmap='gray', origin='lower')
    axes[0, 1].set_title('Sagittal MIP - CT')
    axes[0, 1].axis('off')

    axes[1, 1].imshow(ct_sagittal.T, cmap='gray', origin='lower')
    axes[1, 1].imshow(err_sagittal_t, alpha=alpha, origin='lower')
    axes[1, 1].set_title('Sagittal MIP - CT + Errors')
    axes[1, 1].axis('off')

    # Coronal view
    axes[0, 2].imshow(ct_coronal.T, cmap='gray', origin='lower')
    axes[0, 2].set_title('Coronal MIP - CT')
    axes[0, 2].axis('off')

    axes[1, 2].imshow(ct_coronal.T, cmap='gray', origin='lower')
    axes[1, 2].imshow(err_coronal_t, alpha=alpha, origin='lower')
    axes[1, 2].set_title('Coronal MIP - CT + Errors')
    axes[1, 2].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def render_error_overlay(ct_volume: np.ndarray,
                         error_masks: dict,
                         spacing: tuple,
                         output_path: str,
                         use_mip: bool = True):
    """
    Main function to render error visualization.

    This is a convenience wrapper that creates error RGB map and renders it.

    Args:
        ct_volume: 3D CT volume (H, W, D)
        error_masks: Dictionary from compute_error_masks()
        spacing: Voxel spacing in mm (sx, sy, sz)
        output_path: Path to save PNG
        use_mip: If True, use MIP visualization; otherwise use middle slices
    """
    # Create RGB error map
    error_rgb = create_error_rgb_map(error_masks)

    if use_mip:
        # Use Maximum Intensity Projection
        visualize_error_mip(ct_volume, error_rgb, output_path=output_path, alpha=0.7)
    else:
        # Use middle slice
        H, W, D = ct_volume.shape
        mid_slice = D // 2

        ct_slice = ct_volume[:, :, mid_slice]
        error_slice = error_rgb[:, :, mid_slice]

        visualize_error_slice(ct_slice, error_slice, mid_slice,
                             view='axial', output_path=output_path, alpha=0.7)


def get_largest_error_components(error_masks: dict, n: int = 5) -> dict:
    """
    Identify the N largest false positive and false negative components.

    Args:
        error_masks: Dictionary from compute_error_masks()
        n: Number of largest components to return

    Returns:
        Dictionary with:
            - fp_largest_sizes: List of N largest FP component sizes (voxels)
            - fn_largest_sizes: List of N largest FN component sizes (voxels)
            - fp_largest_centroids: List of N FP component centroids (x, y, z)
            - fn_largest_centroids: List of N FN component centroids (x, y, z)
    """
    # False positives
    fp_labeled, fp_num_components = ndimage.label(error_masks['false_positive'])
    fp_sizes = []
    fp_centroids = []

    for i in range(1, min(fp_num_components + 1, n + 1)):
        size = np.sum(fp_labeled == i)
        centroid = ndimage.center_of_mass(fp_labeled == i)
        fp_sizes.append(size)
        fp_centroids.append(centroid)

    # Sort by size
    if fp_sizes:
        sorted_indices = np.argsort(fp_sizes)[::-1]
        fp_sizes = [fp_sizes[i] for i in sorted_indices]
        fp_centroids = [fp_centroids[i] for i in sorted_indices]

    # False negatives
    fn_labeled, fn_num_components = ndimage.label(error_masks['false_negative'])
    fn_sizes = []
    fn_centroids = []

    for i in range(1, min(fn_num_components + 1, n + 1)):
        size = np.sum(fn_labeled == i)
        centroid = ndimage.center_of_mass(fn_labeled == i)
        fn_sizes.append(size)
        fn_centroids.append(centroid)

    # Sort by size
    if fn_sizes:
        sorted_indices = np.argsort(fn_sizes)[::-1]
        fn_sizes = [fn_sizes[i] for i in sorted_indices]
        fn_centroids = [fn_centroids[i] for i in sorted_indices]

    return {
        'fp_largest_sizes': fp_sizes,
        'fn_largest_sizes': fn_sizes,
        'fp_largest_centroids': fp_centroids,
        'fn_largest_centroids': fn_centroids,
    }

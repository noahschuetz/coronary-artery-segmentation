"""
Uncertainty and Confidence Visualization

This module computes and visualizes prediction uncertainty from softmax probabilities,
helping identify regions where the model is uncertain.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def compute_entropy_map(probabilities: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Compute voxel-wise prediction entropy (uncertainty).

    Higher entropy indicates higher uncertainty.

    Args:
        probabilities: Softmax probabilities (num_classes, H, W, D)
        epsilon: Small constant to avoid log(0)

    Returns:
        Entropy map (H, W, D) with values in [0, -log(1/num_classes)]
    """
    # Add epsilon to avoid log(0)
    probs_safe = probabilities + epsilon

    # Compute Shannon entropy: H(p) = -Σ p_i * log(p_i)
    entropy = -np.sum(probs_safe * np.log(probs_safe), axis=0)

    return entropy


def compute_confidence_map(probabilities: np.ndarray, pred_class: np.ndarray) -> np.ndarray:
    """
    Extract probability of predicted class at each voxel.

    Args:
        probabilities: Softmax probabilities (num_classes, H, W, D)
        pred_class: Predicted class labels (H, W, D)

    Returns:
        Confidence map (H, W, D) with values in [0, 1]
    """
    # Get shape
    H, W, D = pred_class.shape

    # Create confidence map
    confidence = np.zeros((H, W, D), dtype=np.float32)

    # Extract probability for predicted class at each voxel
    for c in range(probabilities.shape[0]):
        mask = (pred_class == c)
        confidence[mask] = probabilities[c, mask]

    return confidence


def compute_margin_map(probabilities: np.ndarray) -> np.ndarray:
    """
    Compute margin between top-2 class probabilities.

    Lower margin indicates higher uncertainty.

    Args:
        probabilities: Softmax probabilities (num_classes, H, W, D)

    Returns:
        Margin map (H, W, D) with values in [0, 1]
    """
    # Sort probabilities along class dimension
    sorted_probs = np.sort(probabilities, axis=0)

    # Margin = difference between top-2 classes
    margin = sorted_probs[-1] - sorted_probs[-2]

    return margin


def create_uncertainty_colormap():
    """
    Create a blue-to-red colormap for uncertainty visualization.

    Blue = low uncertainty (confident)
    Red = high uncertainty (uncertain)

    Returns:
        Matplotlib colormap
    """
    colors = [
        (0.0, 0.0, 1.0),  # Blue (low uncertainty)
        (0.0, 1.0, 1.0),  # Cyan
        (0.0, 1.0, 0.0),  # Green
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 0.0),  # Red (high uncertainty)
    ]

    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('uncertainty', colors, N=n_bins)

    return cmap


def visualize_uncertainty_slice(ct_slice: np.ndarray,
                                uncertainty_slice: np.ndarray,
                                slice_idx: int,
                                view: str = 'axial',
                                output_path: str = None,
                                cmap: str = 'hot',
                                alpha: float = 0.7):
    """
    Visualize uncertainty as an overlay on a CT slice.

    Args:
        ct_slice: 2D CT slice
        uncertainty_slice: 2D uncertainty map
        slice_idx: Slice index for labeling
        view: View plane ('axial', 'sagittal', 'coronal')
        output_path: Path to save figure (optional)
        cmap: Colormap for uncertainty (default: 'hot')
        alpha: Transparency of uncertainty overlay
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # CT alone
    axes[0].imshow(ct_slice, cmap='gray', origin='lower')
    axes[0].set_title(f'{view.capitalize()} View - CT\n(Slice {slice_idx})')
    axes[0].axis('off')

    # Uncertainty alone
    im1 = axes[1].imshow(uncertainty_slice, cmap=cmap, origin='lower')
    axes[1].set_title(f'Uncertainty Map\n(Higher = More Uncertain)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # CT + Uncertainty overlay
    axes[2].imshow(ct_slice, cmap='gray', origin='lower')
    im2 = axes[2].imshow(uncertainty_slice, cmap=cmap, alpha=alpha, origin='lower')
    axes[2].set_title(f'CT + Uncertainty Overlay')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Leave space for colorbar

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_uncertainty_mip(ct_volume: np.ndarray,
                              uncertainty_map: np.ndarray,
                              output_path: str = None,
                              cmap: str = 'hot',
                              alpha: float = 0.7):
    """
    Visualize uncertainty using Maximum Intensity Projection (MIP).

    Creates 3-view MIP (axial, sagittal, coronal) with uncertainty overlay.

    Args:
        ct_volume: 3D CT volume (H, W, D)
        uncertainty_map: 3D uncertainty map (H, W, D)
        output_path: Path to save figure (optional)
        cmap: Colormap for uncertainty
        alpha: Transparency of uncertainty overlay
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Compute MIPs
    ct_axial = np.max(ct_volume, axis=2)
    ct_sagittal = np.max(ct_volume, axis=1)
    ct_coronal = np.max(ct_volume, axis=0)

    unc_axial = np.max(uncertainty_map, axis=2)
    unc_sagittal = np.max(uncertainty_map, axis=1)
    unc_coronal = np.max(uncertainty_map, axis=0)

    # Axial view
    axes[0, 0].imshow(ct_axial, cmap='gray', origin='lower')
    axes[0, 0].set_title('Axial MIP - CT')
    axes[0, 0].axis('off')

    axes[1, 0].imshow(ct_axial, cmap='gray', origin='lower')
    im0 = axes[1, 0].imshow(unc_axial, cmap=cmap, alpha=alpha, origin='lower')
    axes[1, 0].set_title('Axial MIP - CT + Uncertainty')
    axes[1, 0].axis('off')

    # Sagittal view
    axes[0, 1].imshow(ct_sagittal.T, cmap='gray', origin='lower')
    axes[0, 1].set_title('Sagittal MIP - CT')
    axes[0, 1].axis('off')

    axes[1, 1].imshow(ct_sagittal.T, cmap='gray', origin='lower')
    axes[1, 1].imshow(unc_sagittal.T, cmap=cmap, alpha=alpha, origin='lower')
    axes[1, 1].set_title('Sagittal MIP - CT + Uncertainty')
    axes[1, 1].axis('off')

    # Coronal view
    axes[0, 2].imshow(ct_coronal.T, cmap='gray', origin='lower')
    axes[0, 2].set_title('Coronal MIP - CT')
    axes[0, 2].axis('off')

    axes[1, 2].imshow(ct_coronal.T, cmap='gray', origin='lower')
    axes[1, 2].imshow(unc_coronal.T, cmap=cmap, alpha=alpha, origin='lower')
    axes[1, 2].set_title('Coronal MIP - CT + Uncertainty')
    axes[1, 2].axis('off')

    # Add colorbar
    fig.colorbar(im0, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, label='Uncertainty')

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Leave space for colorbar

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def get_high_uncertainty_regions(uncertainty_map: np.ndarray,
                                 threshold_percentile: float = 90) -> np.ndarray:
    """
    Identify regions with high uncertainty.

    Args:
        uncertainty_map: 3D uncertainty map (H, W, D)
        threshold_percentile: Percentile threshold (default: 90th percentile)

    Returns:
        Binary mask of high-uncertainty regions (H, W, D)
    """
    threshold = np.percentile(uncertainty_map, threshold_percentile)
    high_unc_mask = uncertainty_map > threshold

    return high_unc_mask


def compute_uncertainty_statistics(uncertainty_map: np.ndarray, mask: np.ndarray = None) -> dict:
    """
    Compute statistics of uncertainty within a region.

    Args:
        uncertainty_map: 3D uncertainty map (H, W, D)
        mask: Optional binary mask to restrict analysis region

    Returns:
        Dictionary with mean, std, min, max, median uncertainty
    """
    if mask is not None:
        values = uncertainty_map[mask > 0]
    else:
        values = uncertainty_map.flatten()

    stats = {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values),
        'q25': np.percentile(values, 25),
        'q75': np.percentile(values, 75),
    }

    return stats

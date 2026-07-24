"""
Per-Sample Metrics Computation

This module computes evaluation metrics for individual segmentation samples,
enabling identification of best/worst performing cases.
"""

import torch
import numpy as np
import pandas as pd
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric, ConfusionMatrixMetric
from src.metrics.cl_dice import ClDiceMetric


def compute_sample_dice(pred: np.ndarray, label: np.ndarray, include_background: bool = False) -> float:
    """
    Compute Dice coefficient for a single sample.

    Args:
        pred: Binary prediction mask (H, W, D)
        label: Binary ground truth mask (H, W, D)
        include_background: Whether to include background class

    Returns:
        Dice score (float)
    """
    # Convert to torch tensors
    pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)
    label_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W, D)

    # Convert to one-hot encoding
    pred_onehot = torch.zeros((1, 2) + pred.shape, dtype=torch.float32)
    pred_onehot[0, 0] = (pred_t[0, 0] == 0).float()  # background
    pred_onehot[0, 1] = (pred_t[0, 0] == 1).float()  # foreground

    label_onehot = torch.zeros((1, 2) + label.shape, dtype=torch.float32)
    label_onehot[0, 0] = (label_t[0, 0] == 0).float()  # background
    label_onehot[0, 1] = (label_t[0, 0] == 1).float()  # foreground

    # Compute Dice
    metric = DiceMetric(include_background=include_background, reduction="mean")
    dice = metric(y_pred=pred_onehot, y=label_onehot)

    return dice.item()


def compute_sample_iou(pred: np.ndarray, label: np.ndarray, include_background: bool = False) -> float:
    """
    Compute Intersection over Union for a single sample.

    Args:
        pred: Binary prediction mask (H, W, D)
        label: Binary ground truth mask (H, W, D)
        include_background: Whether to include background class

    Returns:
        IoU score (float)
    """
    # Convert to torch tensors
    pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
    label_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)

    # Convert to one-hot encoding
    pred_onehot = torch.zeros((1, 2) + pred.shape, dtype=torch.float32)
    pred_onehot[0, 0] = (pred_t[0, 0] == 0).float()
    pred_onehot[0, 1] = (pred_t[0, 0] == 1).float()

    label_onehot = torch.zeros((1, 2) + label.shape, dtype=torch.float32)
    label_onehot[0, 0] = (label_t[0, 0] == 0).float()
    label_onehot[0, 1] = (label_t[0, 0] == 1).float()

    # Compute IoU
    metric = MeanIoU(include_background=include_background, reduction="mean")
    iou = metric(y_pred=pred_onehot, y=label_onehot)

    return iou.item()


def compute_sample_hausdorff(pred: np.ndarray, label: np.ndarray, percentile: float = 95) -> float:
    """
    Compute Hausdorff distance for a single sample.

    Args:
        pred: Binary prediction mask (H, W, D)
        label: Binary ground truth mask (H, W, D)
        percentile: Percentile to use (default: 95th percentile)

    Returns:
        Hausdorff distance (float)
    """
    # Convert to torch tensors
    pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
    label_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()

    # Compute Hausdorff
    try:
        metric = HausdorffDistanceMetric(include_background=False, percentile=percentile)
        hd = metric(y_pred=pred_t, y=label_t)

        # Handle None or inf return
        if hd is None or torch.isinf(hd) or torch.isnan(hd):
            return 0.0

        return hd.item()
    except:
        # Return 0 if computation fails (empty masks)
        return 0.0


def compute_sample_cldice(pred: np.ndarray, label: np.ndarray, smooth: bool = False) -> float:
    """
    Compute centerline Dice (clDice) for a single sample.

    Args:
        pred: Binary prediction mask (H, W, D)
        label: Binary ground truth mask (H, W, D)
        smooth: Whether to use smooth clDice (distance-weighted)

    Returns:
        clDice score (float)
    """
    # Convert to torch tensors
    pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0)
    label_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0)

    # Convert to one-hot encoding
    pred_onehot = torch.zeros((1, 2) + pred.shape, dtype=torch.float32)
    pred_onehot[0, 0] = (pred_t[0, 0] == 0).float()
    pred_onehot[0, 1] = (pred_t[0, 0] == 1).float()

    label_onehot = torch.zeros((1, 2) + label.shape, dtype=torch.float32)
    label_onehot[0, 0] = (label_t[0, 0] == 0).float()
    label_onehot[0, 1] = (label_t[0, 0] == 1).float()

    # Compute clDice
    metric = ClDiceMetric(include_background=False, smooth=smooth)
    metric(y_pred=pred_onehot, y=label_onehot)  # Accumulates scores
    cldice = metric.aggregate()  # Get the aggregated score

    return cldice.item()


def compute_confusion_matrix_metrics(pred: np.ndarray, label: np.ndarray) -> dict:
    """
    Compute precision, recall, F1, and accuracy from confusion matrix.

    Args:
        pred: Binary prediction mask (H, W, D)
        label: Binary ground truth mask (H, W, D)

    Returns:
        Dictionary with precision, recall, f1, accuracy (all values 0-1)
    """
    # Compute confusion matrix components
    tp = np.sum((pred == 1) & (label == 1)).astype(float)
    fp = np.sum((pred == 1) & (label == 0)).astype(float)
    tn = np.sum((pred == 0) & (label == 0)).astype(float)
    fn = np.sum((pred == 0) & (label == 1)).astype(float)

    # Compute metrics (with division by zero handling)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'accuracy': float(accuracy)
    }


def compute_volume_ml(mask: np.ndarray, spacing: tuple) -> float:
    """
    Compute volume in milliliters from binary mask.

    Args:
        mask: Binary mask (H, W, D)
        spacing: Voxel spacing in mm (sx, sy, sz)

    Returns:
        Volume in mL
    """
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    num_voxels = np.sum(mask > 0)
    volume_mm3 = num_voxels * voxel_volume_mm3
    volume_ml = volume_mm3 / 1000.0  # Convert mm³ to mL

    return volume_ml


def compute_sample_metrics(pred: np.ndarray, label: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0)) -> dict:
    """
    Compute all evaluation metrics for a single sample.

    Args:
        pred: Binary prediction mask (H, W, D)
        label: Binary ground truth mask (H, W, D)
        spacing: Voxel spacing in mm (sx, sy, sz)

    Returns:
        Dictionary with all metrics:
            - dice: Dice coefficient
            - iou: Intersection over Union
            - cldice: Centerline Dice
            - smooth_cldice: Smooth centerline Dice
            - hausdorff_95: 95th percentile Hausdorff distance
            - precision: Precision
            - recall: Recall (sensitivity)
            - f1: F1 score
            - accuracy: Accuracy
            - pred_volume_ml: Predicted vessel volume in mL
            - gt_volume_ml: Ground truth vessel volume in mL
    """
    metrics = {}

    # Volumetric metrics
    metrics['dice'] = compute_sample_dice(pred, label, include_background=False)
    metrics['iou'] = compute_sample_iou(pred, label, include_background=False)

    # Topological metrics
    metrics['cldice'] = compute_sample_cldice(pred, label, smooth=False)
    metrics['smooth_cldice'] = compute_sample_cldice(pred, label, smooth=True)

    # Distance metric
    metrics['hausdorff_95'] = compute_sample_hausdorff(pred, label, percentile=95)

    # Confusion matrix metrics
    cm_metrics = compute_confusion_matrix_metrics(pred, label)
    metrics.update(cm_metrics)

    # Volume metrics
    metrics['pred_volume_ml'] = compute_volume_ml(pred, spacing)
    metrics['gt_volume_ml'] = compute_volume_ml(label, spacing)

    return metrics


def create_metrics_table(results_list: list) -> pd.DataFrame:
    """
    Create a pandas DataFrame with metrics for all samples.

    Args:
        results_list: List of dicts, each containing 'patient_id' and metric values

    Returns:
        pandas DataFrame sorted by Dice score (descending)
    """
    df = pd.DataFrame(results_list)

    # Sort by Dice score (descending)
    if 'dice' in df.columns:
        df = df.sort_values('dice', ascending=False)

    # Reset index
    df = df.reset_index(drop=True)

    return df


def get_sample_statistics(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics across all samples.

    Args:
        df: DataFrame with metrics for all samples

    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    stats = {}

    metric_columns = ['dice', 'iou', 'cldice', 'smooth_cldice', 'hausdorff_95',
                     'precision', 'recall', 'f1', 'accuracy']

    for metric in metric_columns:
        if metric in df.columns:
            stats[f'{metric}_mean'] = df[metric].mean()
            stats[f'{metric}_std'] = df[metric].std()
            stats[f'{metric}_min'] = df[metric].min()
            stats[f'{metric}_max'] = df[metric].max()

    return stats

"""
clDice and Smooth clDice evaluation metrics.

clDice (centerline Dice) measures topology-preservation by evaluating overlap
between predicted/ground-truth skeletons and the opposing volume masks.
Smooth clDice additionally weights skeleton-volume overlap by distance maps.
"""

import torch
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

class ClDiceMetric:
    """
    Computes clDice and Smooth clDice metrics.
    
    Optimized with efficient projection-based ROI cropping and reduced data transfer.
    """
    def __init__(self, include_background=False, smooth=False, alpha=1.0):
        self.include_background = include_background
        self.smooth = smooth
        self.alpha = alpha
        self.scores = []

    def reset(self):
        self.scores = []

    def _get_bbox(self, mask):
        """
        Efficiently calculate bounding box of a 3D binary mask using projections.
        Much faster than np.argwhere for large volumes.
        """
        # Project along axes to find non-zero indices (vectorized)
        z_any = np.any(mask, axis=(1, 2))
        y_any = np.any(mask, axis=(0, 2))
        x_any = np.any(mask, axis=(0, 1))
        
        # If empty, return None
        if not np.any(z_any): return None
        
        # Find first and last True index
        z_min, z_max = np.where(z_any)[0][[0, -1]]
        y_min, y_max = np.where(y_any)[0][[0, -1]]
        x_min, x_max = np.where(x_any)[0][[0, -1]]
        
        return (z_min, z_max, y_min, y_max, x_min, x_max)

    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        Args:
            y_pred: (B, C, Spatial...) One-hot predictions
            y: (B, C, Spatial...) One-hot targets
        """
        # Optimization 1: Convert to boolean on GPU to reduce CPU transfer size
        # (float32 -> bool is 32x smaller payload)
        y_pred_bool = (y_pred > 0.5).bool()
        y_true_bool = (y > 0.5).bool()
        
        # Move to CPU
        y_pred_np = y_pred_bool.cpu().numpy()
        y_true_np = y_true_bool.cpu().numpy()
        
        batch_size, n_class = y_pred_np.shape[:2]
        start_c = 0 if self.include_background else 1
        
        for b in range(batch_size):
            class_scores = []
            
            for c in range(start_c, n_class):
                p = y_pred_np[b, c]
                t = y_true_np[b, c]
                
                # Optimization 2: Fast emptiness check
                if not np.any(p) and not np.any(t):
                    class_scores.append(1.0)
                    continue
                if not np.any(p) or not np.any(t):
                    class_scores.append(0.0)
                    continue

                # Optimization 3: Efficient ROI Cropping via Projection
                # Union mask to find common ROI
                mask = p | t
                bbox = self._get_bbox(mask)
                
                if bbox is None:
                    class_scores.append(1.0)
                    continue
                    
                z1, z2, y1, y2, x1, x2 = bbox
                
                # Add padding
                pad = 5
                z1 = max(0, z1 - pad); z2 = min(p.shape[0], z2 + 1 + pad)
                y1 = max(0, y1 - pad); y2 = min(p.shape[1], y2 + 1 + pad)
                x1 = max(0, x1 - pad); x2 = min(p.shape[2], x2 + 1 + pad)
                
                p_crop = p[z1:z2, y1:y2, x1:x2]
                t_crop = t[z1:z2, y1:y2, x1:x2]

                # Skeletonization on ROI-cropped volume
                try:
                    s_p = skeletonize(p_crop)
                    s_t = skeletonize(t_crop)
                except ValueError:
                    class_scores.append(0.0)
                    continue

                # Metric Calculation
                if self.smooth:
                    # Invert for distance transform (distance to nearest False)
                    # Since inputs are bool, ~ works correctly
                    dist_t = distance_transform_edt(~t_crop)
                    d_t = np.exp(-self.alpha * dist_t)
                    
                    dist_p = distance_transform_edt(~p_crop)
                    d_p = np.exp(-self.alpha * dist_p)
                    
                    t_prec = np.sum(s_p * d_t) / (np.sum(s_p) + 1e-6)
                    t_sens = np.sum(s_t * d_p) / (np.sum(s_t) + 1e-6)
                else:
                    t_prec = np.sum(s_p * t_crop) / (np.sum(s_p) + 1e-6)
                    t_sens = np.sum(s_t * p_crop) / (np.sum(s_t) + 1e-6)

                cldice = 2.0 * t_prec * t_sens / (t_prec + t_sens + 1e-6)
                class_scores.append(cldice)
            
            if class_scores:
                self.scores.append(np.mean(class_scores))
            else:
                self.scores.append(0.0)

    def aggregate(self):
        if not self.scores:
            return torch.tensor(0.0)
        return torch.tensor(np.mean(self.scores))
    
    def get_buffer(self):
        # For compatibility with MONAI logging in train.py
        return [torch.tensor(s) for s in self.scores]

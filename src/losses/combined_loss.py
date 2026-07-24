"""
Combined DiceCE + clDice loss for topology-preserving vessel segmentation.

DiceCELoss provides voxel-level accuracy while SoftCLDiceLoss enforces
centerline connectivity — critical for thin tubular structures like
coronary arteries.
"""

import torch
import torch.nn as nn
from monai.losses import DiceCELoss
from src.losses.soft_cldice_loss import SoftCLDiceLoss


class DiceCECLDiceLoss(nn.Module):
    """Combined DiceCE + clDice loss.

    Args:
        lambda_dice_ce: Weight for the DiceCE component.
        lambda_cldice: Weight for the clDice component.
        cldice_iter: Number of soft-skeleton erosion iterations.
    """

    def __init__(self, lambda_dice_ce=0.7, lambda_cldice=0.3, cldice_iter=3):
        super().__init__()
        self.dice_ce = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            lambda_dice=0.8,
            lambda_ce=0.2,
        )
        self.cldice = SoftCLDiceLoss(iter_=cldice_iter)
        self.lambda_dice_ce = lambda_dice_ce
        self.lambda_cldice = lambda_cldice

    def forward(self, logits, labels):
        loss_dice_ce = self.dice_ce(logits, labels)

        # clDice needs softmax probabilities + one-hot labels
        probs = torch.softmax(logits, dim=1)
        labels_oh = torch.zeros_like(probs)
        labels_oh.scatter_(1, labels.long(), 1.0)
        loss_cldice = self.cldice(probs, labels_oh)

        return self.lambda_dice_ce * loss_dice_ce + self.lambda_cldice * loss_cldice

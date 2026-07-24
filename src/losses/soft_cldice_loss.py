"""
Soft clDice loss for topology-preserving segmentation training.

Reference: Shit et al., "clDice - a Novel Topology-Preserving Loss Function
for Tubular Structure Segmentation", CVPR 2021.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftCLDiceLoss(nn.Module):
    """
    Soft clDice Loss (Shit et al. CVPR 2021).
    
    This is a differentiable loss function for training that approximates 
    the skeletonization process. It is distinct from the "Smooth clDice" 
    metric (2025) which uses distance maps for evaluation.
    """
    def __init__(self, iter_=3, smooth=1.0):
        super(SoftCLDiceLoss, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B, C, X, Y, Z) Softmax probabilities
            y_true: (B, C, X, Y, Z) One-hot ground truth
        """
        # Focus on the foreground class (assuming index 1 is vessels)
        # If you have multiple foregrounds, loop over them
        if y_pred.shape[1] > 1:
            y_pred = y_pred[:, 1:2]
            y_true = y_true[:, 1:2]
            
        skel_pred = self.soft_skel(y_pred, self.iter)
        skel_true = self.soft_skel(y_true, self.iter)
        
        # T_prec = |S_P \cap V_L| / |S_P|
        t_prec = (torch.sum(skel_pred * y_true) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
        
        # T_sens = |S_L \cap V_P| / |S_L|
        t_sens = (torch.sum(skel_true * y_pred) + self.smooth) / (torch.sum(skel_true) + self.smooth)
        
        cl_dice = 2.0 * (t_prec * t_sens) / (t_prec + t_sens)
        return 1.0 - cl_dice

    def soft_erode(self, img):
        if len(img.shape) == 4: # 2D
            p1 = -F.max_pool2d(-img, (3,1), (1,1), (1,0))
            p2 = -F.max_pool2d(-img, (1,3), (1,1), (0,1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5: # 3D
            p1 = -F.max_pool3d(-img, (3,1,1), (1,1,1), (1,0,0))
            p2 = -F.max_pool3d(-img, (1,3,1), (1,1,1), (0,1,0))
            p3 = -F.max_pool3d(-img, (1,1,3), (1,1,1), (0,0,1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):
        if len(img.shape) == 4:
            return F.max_pool2d(img, (3,3), (1,1), (1,1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3,3,3), (1,1,1), (1,1,1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img, iter_):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)
        for i in range(iter_):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

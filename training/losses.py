import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self,logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)


        intersection = (probs * targets).sum()
        dice_coeff = (2.0*intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice_coeff
    


class DiceBCELoss(nn.Module):
    def __init__(self,bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self,logits, targets):
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits,targets)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

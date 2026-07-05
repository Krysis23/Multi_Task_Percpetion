import torch

def iou_score(logits,targets, threshold: float=0.5, smooth:float= 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection  = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return ((intersection + smooth)/ (union + smooth)).item()

def dice_coefficient(logits,targets,threshold: float=0.5, smooth:float=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection =(preds * targets).sum()
    return ((0.2 * intersection + smooth)/ (preds.sum() + targets.sum() + smooth)).item()



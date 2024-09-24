import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = logpt.exp()
        pt = pt.clamp(min=1e-8, max=1.0)
        loss = -self.alpha * (1 - pt) ** self.gamma * logpt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

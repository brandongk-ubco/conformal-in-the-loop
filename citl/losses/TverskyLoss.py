import torch
import torch.nn.functional as F


class TverskyLoss(torch.nn.Module):
    def __init__(
        self,
        smooth=1e-6,
        from_logits=True,
        gamma=2.0,
        eps: float = 1e-6,
    ):
        super(TverskyLoss, self).__init__()
        self.eps = eps
        self.smooth = smooth
        self.from_logits = from_logits
        self.gamma = gamma

    def forward(self, inputs, targets):

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        alpha = 0.5

        if self.from_logits:
            logistic_inputs = torch.sigmoid(inputs)
        else:
            logistic_inputs = inputs

        # True Positives, False Positives & False Negatives
        TP = (logistic_inputs * targets).sum()
        FP = ((1 - targets) * logistic_inputs).sum()
        FN = (targets * (1 - logistic_inputs)).sum()

        tversky = (TP + self.smooth) / (
            TP + 2 * alpha * FP + 2 * (1 - alpha) * FN + self.smooth
        )

        loss = 1 - tversky

        return loss**self.gamma

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithLogits():
    def __init__(self, gamma=1):
        self.gamma = gamma
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, outputs, labels, reduction='mean'):
        focal_weight = ((1 - outputs).sigmoid() * labels + outputs.sigmoid() * (1 - labels)) ** self.gamma
        loss = self.loss_fn(outputs, labels) * focal_weight

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise NotImplementedError


class MultiClassFocalLossWithLogits(nn.Module):
    def __init__(self, device, alpha=0.25, gamma=2):
        super(MultiClassFocalLossWithLogits, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def forward(self, inputs, targets, reduction='mean'):

        num_classes = inputs.size(1)
        weight = torch.Tensor([self.alpha] + [1 - self.alpha] * (num_classes - 1)).to(self.device)

        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none', weight=weight)
        pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma * ce_loss)

        if reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'sum':
            return focal_loss.sum()
        elif reduction == 'none':
            return focal_loss
        else:
            raise NotImplementedError

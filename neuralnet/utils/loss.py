import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, predicted, targets):
        smooth = 1
        num = targets.size(0)
        m1 = predicted.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = (5. * intersection.sum(1) + smooth) / (4 * ((m1.sum(1) + m2.sum(1)) + smooth))
        score = 1 - score.sum() / num
        return score


dice_loss = SoftDiceLoss()

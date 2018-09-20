import torch
import torch.nn as nn

from torch.autograd import Function, Variable


class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, predicted, targets):
        smooth = 1
        num = targets.size(0)
        m1 = predicted.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = (2. * intersection.sum(1) + smooth) / ((m1.sum(1) + m2.sum(1)) + smooth)
        score = 1 - score.sum() / num
        return score


# dice_loss = DiceCoeff()


class WeightedMSE(nn.Module):
    def __init__(self):
        super(WeightedMSE, self).__init__()

    def forward(self, inputs, targets, weights):
        out = (inputs - targets) ** 2
        out = out * weights.expand_as(out)
        loss = torch.mean(out)  # or sum over whatever dimensions
        return loss


weighted_mse_loss = WeightedMSE()


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union + self.inter) \
                         / self.union * self.union
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

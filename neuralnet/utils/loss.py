import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        self.inter = torch.dot(input.contiguous().view(-1), target.contiguous().view(-1).float()) + 0.0001
        self.union = torch.sum(input) + torch.sum(target.float()) + 0.0001

        t = 2 * self.inter.float() / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output.long() * 2 * (target.long() * self.union.long() + self.inter.long()) \
                         / self.union.long() * self.union.long()
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_loss(input, target):
    return DiceCoeff()(input, target)

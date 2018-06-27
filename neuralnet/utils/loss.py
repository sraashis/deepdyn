import torch


def dice_loss(input, target):
    smooth = 1.
    iflat = input.view(-1).cuda()
    tflat = target.view(-1).cuda()
    intersection = (iflat * tflat).sum()

    # print(intersection, iflat.sum(), tflat.sum())
    return 1 - ((5. * intersection + smooth) /
                (4. * iflat.sum() + tflat.sum() + smooth))


def shift_loss(input):
    r = torch.cat((input[:, 0:1], input[:, :-1]), 1)  ### Right
    l = torch.cat((input[:, 1:], input[:, -1:]), 1)  ### Left
    d = torch.cat((input[0:1, :], input[:-1, :]), 0)  ### Down
    u = torch.cat((input[1:, :], input[-1:, :]), 0)  ### Left
    shifted = l + r + d + u + input
    return shifted / 5

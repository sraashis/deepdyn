def dice_loss(outputs, target, beta=1):
    smooth = 1.

    iflat = outputs.contiguous().float().view(-1)
    tflat = target.contiguous().float().view(-1)
    intersection = (iflat * tflat).sum()

    loss = (((1 + beta ** 2) * intersection) + smooth) / (((beta ** 2 + iflat.sum()) + tflat.sum()) + smooth)
    return 1 - loss

def dice_loss(outputs=None, target=None, beta=1, weights=None, ):
    """
    :param weights: element-wise weights
    :param outputs:
    :param target:
    :param beta: More beta, better precision. 1 is neutral
    :return:
    """
    from torch import min as tmin
    smooth = 1.0
    if weights is not None:
        weights = weights.contiguous().float().view(-1)
        if tmin(weights).item() == 0:
            weights += smooth
    else:
        weights = 1

    iflat = outputs.contiguous().float().view(-1)
    tflat = target.contiguous().float().view(-1)
    intersection = (iflat * tflat * weights).sum()

    f = (((1 + beta ** 2) * intersection) + smooth) / (
            ((beta ** 2 * (weights * iflat).sum()) + (weights * tflat).sum()) + smooth)
    return 1 - f
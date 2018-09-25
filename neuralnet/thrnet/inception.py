import itertools

import torch
import torch.nn.functional as F
from torch import nn

from neuralnet.utils.weights_utils import initialize_weights


class BasicConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, k, s, p):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(self, in_ch=None, width=None, out_ch=128):
        super(Inception, self).__init__()

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=3)
        self.convA_3by3 = BasicConv2d(in_ch=in_ch, out_ch=int(out_ch / 2), k=k, s=s, p=p)

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=3)
        self.convB_3by3 = BasicConv2d(in_ch=in_ch, out_ch=int(out_ch), k=k, s=s, p=p)

        self.convB_out_1by1 = BasicConv2d(in_ch=out_ch, out_ch=int(out_ch / 2), k=1, s=1, p=0)

    def forward(self, x):
        a = self.convA_3by3(x)
        b = self.convB_out_1by1(self.convB_3by3(x))
        return torch.cat([a, b], 1)

    @staticmethod
    def out_w(w, k, s, p):
        return ((w - k + 2 * p) / s) + 1

    def get_wksp(self, w=None, w_match=None, k=None, strides=[1, 2, 3], paddings=[0, 1, 2, 3]):
        all_sp = itertools.product(strides, paddings)
        for (s, p) in all_sp:
            w_out = self.out_w(w, k, s, p)
            if w_out.is_integer() and w_match == int(w_out):
                return w_out, k, s, p

        raise LookupError('Solution not within range.')


class InceptionThrNet(nn.Module):
    def __init__(self, width, input_ch, num_class):
        super(InceptionThrNet, self).__init__()

        self.inception1 = Inception(width=width, in_ch=input_ch, out_ch=64)

        self.inception2 = Inception(width=width, in_ch=64, out_ch=128)
        self.inception3 = Inception(width=width, in_ch=128, out_ch=384)
        self.inception4 = Inception(width=width, in_ch=384, out_ch=768)

        self.inception5 = Inception(width=width, in_ch=768, out_ch=256)
        self.inception6 = Inception(width=width, in_ch=256, out_ch=64)
        self.inception7 = nn.Conv2d(in_channels=64, out_channels=num_class, kernel_size=1, stride=1, padding=0)

        initialize_weights(self)

    def forward(self, x):
        i1_out = self.inception1(x)

        i2_out = self.inception2(i1_out)
        i3_out = self.inception3(i2_out)
        i4_out = self.inception4(i3_out)

        i5_out = self.inception5(i4_out)
        i6_out = self.inception6(i5_out)
        i7_out = self.inception7(i6_out)
        return F.log_softmax(i7_out, dim=1)


m = InceptionThrNet(width=64, input_ch=1, num_class=2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)

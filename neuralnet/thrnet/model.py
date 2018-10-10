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
    def __init__(self, in_ch=None, width=None, out_ch=128, k=1):
        super(Inception, self).__init__()

        _, _, s, p = self.get_wksp(w=width, w_match=width, k=k)
        self.convA = BasicConv2d(in_ch=in_ch, out_ch=int(out_ch / 4), k=k, s=s, p=p)

        _, _, s, p = self.get_wksp(w=width, w_match=width, k=k)
        self.convB = BasicConv2d(in_ch=in_ch, out_ch=int(out_ch / 4), k=k, s=s, p=p)

        _, _, s, p = self.get_wksp(w=width, w_match=width, k=k)
        self.convC = BasicConv2d(in_ch=in_ch, out_ch=int(out_ch / 4), k=k, s=s, p=p)

        _, _, s, p = self.get_wksp(w=width, w_match=width, k=k)
        self.convD = BasicConv2d(in_ch=in_ch, out_ch=int(out_ch / 4), k=k, s=s, p=p)

    def forward(self, x):
        a = self.convA(x)
        b = self.convB(x)
        c = self.convC(x)
        d = self.convD(x)
        return torch.cat([a, b, c, d], 1)

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
    def __init__(self, input_ch, num_class):
        super(InceptionThrNet, self).__init__()

        self.inception1 = Inception(width=48, in_ch=input_ch, out_ch=128, k=3)
        self.inception1a = Inception(width=48, in_ch=128, out_ch=256, k=1)

        self.inception2 = Inception(width=40, in_ch=256, out_ch=256, k=3)
        self.inception2a = Inception(width=40, in_ch=256, out_ch=256, k=1)

        self.inception3 = Inception(width=32, in_ch=512, out_ch=512, k=3)
        self.mxp_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inception4 = Inception(width=16, in_ch=512, out_ch=384, k=1)
        self.mxp_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inception5 = Inception(width=8, in_ch=384, out_ch=256, k=3)
        self.mxp_5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inception6 = Inception(width=4, in_ch=256, out_ch=128, k=1)
        self.mxp_6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.out_conv = BasicConv2d(in_ch=128, out_ch=32, k=1, s=1, p=0)

        self.fc1 = nn.Linear(32 * 2 * 2, num_class)
        initialize_weights(self)

    def forward(self, x):
        i1_out = self.inception1(x)
        i1_out = self.inception1a(i1_out)

        i2_out = self.inception2(i1_out[:, :, 4:44, 4:44])
        i2_out = self.inception2a(i2_out)

        i3_out = self.inception3(torch.cat([i1_out[:, :, 8:40, 8:40], i2_out[:, :, 4:36, 4:36]], 1))
        i3_out = self.mxp_3(i3_out)

        i4_out = self.inception4(i3_out)
        i4_out = self.mxp_4(i4_out)

        i5_out = self.inception5(i4_out)
        i5_out = self.mxp_5(i5_out)

        i6_out = self.inception6(i5_out)
        i6_out = self.mxp_6(i6_out)
        conv_out = self.out_conv(i6_out)

        flat = conv_out.view(-1, 32 * 2 * 2)
        return self.fc1(flat)


m = InceptionThrNet(input_ch=1, num_class=1)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)

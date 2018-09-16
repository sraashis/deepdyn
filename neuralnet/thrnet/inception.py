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
        return F.elu(x, inplace=True)


class Inception(nn.Module):
    def __init__(self, in_ch=None, width=None, out_ch=128):
        super(Inception, self).__init__()
        self.width = width
        self.out_ch_per_cell = int(out_ch / 4)

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=1)
        self.convA_in_1by1 = BasicConv2d(in_ch=in_ch, out_ch=self.out_ch_per_cell, k=k, s=s, p=p)

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=1)
        self.convB_in_1by1 = BasicConv2d(in_ch=in_ch, out_ch=self.out_ch_per_cell, k=k, s=s, p=p)

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=1)
        self.convC_in_1by1 = BasicConv2d(in_ch=in_ch, out_ch=self.out_ch_per_cell, k=k, s=s, p=p)

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=3)
        self.convD_in_3by3 = BasicConv2d(in_ch=in_ch, out_ch=self.out_ch_per_cell, k=k, s=s, p=p)

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=3)
        self.convB_3by3 = BasicConv2d(in_ch=self.out_ch_per_cell, out_ch=self.out_ch_per_cell, k=k, s=s, p=p)
        _, k, s, p = self.get_wksp(w=width, w_match=width, k=5)
        self.convC_5by5 = BasicConv2d(in_ch=self.out_ch_per_cell, out_ch=self.out_ch_per_cell, k=k, s=s, p=p)

    def forward(self, x):
        a = self.convA_in_1by1(x)
        b = self.convB_3by3(self.convB_in_1by1(x))
        c = self.convC_5by5(self.convC_in_1by1(x))
        d = self.convD_in_3by3(x)
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


class InceptionRecursiveDownSample(nn.Module):
    def __init__(self, width, in_ch, out_ch, recursion=1):
        super(InceptionRecursiveDownSample, self).__init__()
        layers = []
        for i in range(recursion):
            inception = Inception(width=width, in_ch=in_ch, out_ch=out_ch)
            layers.append(inception)
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            width = width / 2
            in_ch = out_ch
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class InceptionThrNet(nn.Module):
    def __init__(self, width, input_ch, num_class):
        super(InceptionThrNet, self).__init__()

        self.inception1 = Inception(width=width, in_ch=input_ch, out_ch=256)
        self.inception2 = Inception(width=width, in_ch=256, out_ch=256)

        self.inception2_dwn = InceptionRecursiveDownSample(width=width, in_ch=256, out_ch=256, recursion=1)

        self.inception3 = Inception(width=width, in_ch=512, out_ch=1024)
        self.inception4 = Inception(width=width, in_ch=1024, out_ch=512)

        self.inception4_dwn = InceptionRecursiveDownSample(width=width, in_ch=512, out_ch=256, recursion=2)

        self.linearWidth = 256 * 4 * 4
        self.fc1_out = nn.Linear(self.linearWidth, 256)
        self.fc2_out = nn.Linear(256, num_class)
        initialize_weights(self)

    def forward(self, x):
        i1_out = self.inception1(x)
        i2_out = self.inception2(i1_out)

        i2_out_dwn = self.inception2_dwn(i2_out)

        i3_out = self.inception3(torch.cat([i2_out[:, :, 8:24, 8:24], i2_out_dwn], 1))
        i4_out = self.inception4(i3_out)
        i4_dwn_out = self.inception4_dwn(i4_out)

        flattened = i4_dwn_out.view(-1, self.linearWidth)
        fc1_out = self.fc1_out(F.elu(flattened, inplace=True))
        fc2_out = self.fc2_out(fc1_out)

        return fc2_out


import numpy as np

i = InceptionThrNet(width=32, input_ch=1, num_class=1)
model_parameters = filter(lambda p: p.requires_grad, i.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
# print(i)

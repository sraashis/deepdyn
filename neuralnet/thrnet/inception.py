import itertools
import math

import torch
import torch.nn.functional as F
from torch import nn

from neuralnet.utils.weights_utils import initialize_weights


class BasicConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, k, s, p):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):
    def __init__(self, in_ch=None, width=None, out_ch=128):
        super(Inception, self).__init__()
        self.width = width
        self.out_ch_per_cell = int(out_ch / 4)

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=1)
        self.conv_in_1by1 = BasicConv2d(in_ch=in_ch, out_ch=self.out_ch_per_cell, k=k, s=s, p=p)

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=3)
        self.conv_in_3by3 = BasicConv2d(in_ch=in_ch, out_ch=self.out_ch_per_cell, k=k, s=s, p=p)

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=3)
        self.convB_3by3 = BasicConv2d(in_ch=self.out_ch_per_cell, out_ch=self.out_ch_per_cell, k=k, s=s, p=p)
        _, k, s, p = self.get_wksp(w=width, w_match=width, k=5)
        self.convC_5by5 = BasicConv2d(in_ch=self.out_ch_per_cell, out_ch=self.out_ch_per_cell, k=k, s=s, p=p)

    def forward(self, x):
        conv_in_1by1 = self.conv_in_1by1(x)
        convB_3by3_out = self.convB_3by3(conv_in_1by1.clone())
        convC_5by5_out = self.convC_5by5(conv_in_1by1.clone())
        conv_3by3_out = self.conv_in_3by3(x)

        inception_cat = torch.cat([conv_in_1by1, convB_3by3_out, convC_5by5_out, conv_3by3_out], 1)
        return inception_cat

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
    def __init__(self, width, in_ch, out_ch):
        super(InceptionRecursiveDownSample, self).__init__()
        layers = []
        for i in range(1, int(math.log2(width) - 1)):
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
        self.inception1_rec = InceptionRecursiveDownSample(width=width, in_ch=256, out_ch=256)

        self.inception2 = Inception(width=width, in_ch=256, out_ch=1024)
        self.inception2_rec = InceptionRecursiveDownSample(width=width, in_ch=1024, out_ch=256)

        self.inception3 = Inception(width=width, in_ch=1024, out_ch=256)
        self.inception3_rec = InceptionRecursiveDownSample(width=width, in_ch=256, out_ch=256)

        self.conv = BasicConv2d(in_ch=256 * 3, out_ch=64, k=1, s=1, p=0)
        self.linearWidth = 64 * 4 * 4
        self.fc = nn.Linear(self.linearWidth, num_class)

        initialize_weights(self)

    def forward(self, x):
        i1_out = self.inception1(x)
        i1_rec_out = self.inception1_rec(i1_out)

        i2_out = self.inception2(i1_out)
        i2_rec_out = self.inception2_rec(i2_out)
        i2_out = F.dropout2d(i2_out, p=0.2)

        i3_out = self.inception3(i2_out)
        i3_rec_out = self.inception3_rec(i3_out)
        rec_out = torch.cat([i1_rec_out, i2_rec_out, i3_rec_out], 1)
        conv_out = self.conv(rec_out)

        flat = conv_out.view(-1, self.linearWidth)
        flat = F.dropout2d(flat, p=0.2)
        out = self.fc(flat)

        return out

import numpy as np
i = InceptionThrNet(width=64, input_ch=1, num_class=1)
model_parameters = filter(lambda p: p.requires_grad, i.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
# print(i)

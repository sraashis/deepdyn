import itertools
import math

import torch
import torch.nn.functional as F
from torch import nn


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
    def __init__(self, in_ch=None, width=None, out_ch=128, downsample=False):
        super(Inception, self).__init__()
        self.width = width
        self.downsample = downsample

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=1)
        self.conv_in_1by1 = BasicConv2d(in_ch=in_ch, out_ch=out_ch, k=k, s=s, p=p)

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=3)
        self.conv_in_3by3 = BasicConv2d(in_ch=in_ch, out_ch=out_ch, k=k, s=s, p=p)

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=3)
        self.convB_3by3 = BasicConv2d(in_ch=out_ch, out_ch=out_ch, k=k, s=s, p=p)
        _, k, s, p = self.get_wksp(w=width, w_match=width, k=5)
        self.convC_5by5 = BasicConv2d(in_ch=out_ch, out_ch=out_ch, k=k, s=s, p=p)

        self.conv_out_1by1 = BasicConv2d(in_ch=4 * out_ch, out_ch=out_ch, k=k, s=s, p=p)

    def forward(self, x):
        conv_in_1by1 = self.conv_in_1by1(x)

        convB_3by3_out = self.convB_3by3(conv_in_1by1.clone())
        convC_5by5_out = self.convC_5by5(conv_in_1by1.clone())

        conv_3by3_out = self.conv_in_3by3(x)

        if self.downsample:
            conv_in_1by1 = F.max_pool2d(conv_in_1by1, kernel_size=2, stride=2, padding=0)
            convB_3by3_out = F.max_pool2d(convB_3by3_out, kernel_size=2, stride=2, padding=0)
            convC_5by5_out = F.max_pool2d(convC_5by5_out, kernel_size=2, stride=2, padding=0)
            conv_3by3_out = F.max_pool2d(conv_3by3_out, kernel_size=2, stride=2, padding=0)

        inception_cat = torch.cat([conv_in_1by1, convB_3by3_out, convC_5by5_out, conv_3by3_out], 1)
        return self.conv_out_1by1(inception_cat)

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
        for i in range(1, int(math.log2(width)) - 1):
            inception = Inception(width=width, in_ch=in_ch, out_ch=out_ch, downsample=True)
            layers.append(inception)
            width = width / 2
            in_ch = out_ch
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class InceptionThrNet(nn.Module):
    def __init__(self, width, input_ch, num_class):
        super(InceptionThrNet, self).__init__()
        self.inception1_rec = InceptionRecursiveDownSample(width=width, in_ch=input_ch, out_ch=16)
        self.inception1 = Inception(width=width, in_ch=input_ch, out_ch=64)
        self.inception2 = Inception(width=width, in_ch=64, out_ch=64)

        self.inception2_rec = InceptionRecursiveDownSample(width=width, in_ch=64, out_ch=16)
        self.inception3 = Inception(width=width, in_ch=64, out_ch=64)
        self.inception4 = Inception(width=width, in_ch=64, out_ch=64)

        self.inception3_rec = InceptionRecursiveDownSample(width=width, in_ch=64, out_ch=16)
        self.inception5 = Inception(width=width, in_ch=64, out_ch=64)
        self.inception6 = Inception(width=width, in_ch=64, out_ch=64)

        self.inception_rec_all = Inception(width=width, in_ch=16 * 3, out_ch=32)
        self.inception_rec_final = InceptionRecursiveDownSample(width=width, in_ch=64, out_ch=32)

        # concat self.inception_rec_final and self.inception_rec_all
        self.conv = BasicConv2d(in_ch=2 * 32, out_ch=32, k=1, s=1, p=0)

        self.linearWidth = 32 * 4 * 4
        self.fc1 = nn.Linear(self.linearWidth, 32)
        self.out = nn.Linear(32, num_class)

    def forward(self, x):
        i1_rec_out = self.inception1_rec(x)
        i1_out = self.inception1(x)
        i2_out = self.inception2(i1_out)
        i2_out = F.dropout2d(i2_out, p=0.2)

        i2_rec_out = self.inception2_rec(i2_out)
        i3_out = self.inception3(i2_out)
        i4_out = self.inception4(i3_out)
        i4_out = F.dropout2d(i4_out, p=0.2)

        i3_rec_out = self.inception3_rec(i4_out)
        i5_out = self.inception5(i4_out)
        i6_out = self.inception6(i5_out)
        i6_out = F.dropout2d(i6_out, p=0.2)

        rec_final_out = self.inception_rec_final(i6_out)

        full_concat = torch.cat([i1_rec_out, i2_rec_out, i3_rec_out], 1)
        rec_prev = self.inception_rec_all(full_concat)
        final_conv = torch.cat([rec_prev, rec_final_out], 1)
        final_conv = self.conv(final_conv)

        flat = final_conv.view(-1, self.linearWidth)
        f1 = F.relu(self.fc1(flat), inplace=True)
        f1 = F.dropout2d(f1, p=0.2)
        return self.out(f1)


import numpy as np

i = InceptionThrNet(width=64, input_ch=1, num_class=1)
model_parameters = filter(lambda p: p.requires_grad, i.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

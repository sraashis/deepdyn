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
        self.convA1_3by3 = BasicConv2d(in_ch=in_ch, out_ch=out_ch, k=k, s=s, p=p)

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=5)
        self.convB1_5by5 = BasicConv2d(in_ch=in_ch, out_ch=out_ch, k=k, s=s, p=p)

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=5)
        self.convA2_5by5 = BasicConv2d(in_ch=out_ch, out_ch=out_ch, k=k, s=s, p=p)

        _, k, s, p = self.get_wksp(w=width, w_match=width, k=3)
        self.convB2_3by3 = BasicConv2d(in_ch=out_ch, out_ch=out_ch, k=k, s=s, p=p)

        self.conv_out_1by1 = BasicConv2d(in_ch=out_ch * 2, out_ch=out_ch, k=1, s=1, p=0)

    def forward(self, x):
        a = self.convA2_5by5(self.convA1_3by3(x))
        b = self.convB2_3by3(self.convB1_5by5(x))
        return self.conv_out_1by1(torch.cat([a, b], 1))

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

        self.inception1 = Inception(width=width, in_ch=input_ch, out_ch=16)
        self.inception1_mxp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # We will crop and concat from inception1 to this layer
        self.inception2 = Inception(width=width, in_ch=32, out_ch=32)
        self.inception2_mxp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.inception3 = Inception(width=width, in_ch=32, out_ch=32)
        self.inception4 = Inception(width=width, in_ch=32, out_ch=32)
        self.inception4_mxp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.inception5 = Inception(width=width, in_ch=32, out_ch=32)

        self.linearWidth = 32 * 4 * 4
        self.fc1_out = nn.Linear(self.linearWidth, 512)
        self.fc2_out = nn.Linear(512, 64)
        self.fc3_out = nn.Linear(64, num_class)
        initialize_weights(self)

    def forward(self, x):
        i1_out = self.inception1(x)
        i1_out_dwn = self.inception1_mxp(i1_out)

        i2_out = self.inception2(torch.cat([i1_out[:, :, 8:24, 8:24], i1_out_dwn], 1))
        i2_dwn_out = self.inception2_mxp(i2_out)

        i3_out = self.inception3(i2_dwn_out)
        i4_out = self.inception4(i3_out)
        i4_dwn_out = self.inception4_mxp(i4_out)

        i5_out = self.inception5(i4_dwn_out)

        flattened = i5_out.view(-1, self.linearWidth)
        fc1_out = F.relu(self.fc1_out(flattened))
        fc2_out = F.relu(self.fc2_out(fc1_out))

        return self.fc3_out(fc2_out)


m = InceptionThrNet(width=32, input_ch=1, num_class=1)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)

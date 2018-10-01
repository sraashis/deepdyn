def out_w(w, k, s, p):
    return ((w - k + 2 * p) / s) + 1


import torch.nn.functional as F
from torch import nn


class BasicConv2d(nn.Module):

    def __init__(self, in_ch, out_ch, k, s, p):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class PatchNet(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(PatchNet, self).__init__()
        self.conv1 = BasicConv2d(in_ch=in_channels, out_ch=32, k=3, s=1, p=1)
        self.conv2 = BasicConv2d(in_ch=32, out_ch=48, k=5, s=2, p=0)
        self.mxp_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = BasicConv2d(in_ch=48, out_ch=64, k=3, s=1, p=0)
        self.conv4 = BasicConv2d(in_ch=64, out_ch=48, k=3, s=1, p=0)
        self.mxp_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_out = BasicConv2d(in_ch=48, out_ch=32, k=1, s=1, p=0)

        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv2 = self.mxp_2(conv2)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv4 = self.mxp_4(conv4)

        conv_out = self.conv_out(conv4)

        conv_out = conv_out.view(-1, 32 * 4 * 4)
        fc1 = F.relu(self.fc1(conv_out))
        fc2 = F.relu(self.fc2(fc1))
        return F.log_softmax(self.fc_out(fc2), 1)


m = PatchNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(torch_total_params)
# print(out_w(12, 3, 1, 0))

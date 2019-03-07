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


class TrackNet(nn.Module):
    def __init__(self, num_channels, num_class):
        super(TrackNet, self).__init__()
        self.conv1 = BasicConv2d(in_ch=num_channels, out_ch=64, k=3, s=1, p=1)
        o1 = self.output(111, 3, 1, 1)
        # print('o1', o1)
        self.conv2 = BasicConv2d(in_ch=64, out_ch=256, k=3, s=2, p=1)
        o2 = self.output(o1, 3, 2, 1)/2
        # print('o2', o2)
        self.conv3 = BasicConv2d(in_ch=256, out_ch=256, k=3, s=2, p=1)
        o3 = self.output(o2, 3, 1, 1)
        # print('o3', o3)
        self.conv4 = BasicConv2d(in_ch=256, out_ch=128, k=3, s=3, p=1)
        o4 = self.output(o3, 3, 1, 1)/2
        # print('o4', o4)
        self.conv5 = BasicConv2d(in_ch=128, out_ch=64, k=3, s=1, p=1)
        o5 = self.output(o4, 3, 1, 1)
        # print('o5', o5)
        self.linearWidth = 64 * 2 * 2
        self.fc1 = nn.Linear(self.linearWidth, 64)
        self.out = nn.Linear(64, num_class)

    def forward(self, x):
        x = self.conv1(x)
        print('conv1 shape:', x.shape)
        x = self.conv2(x)
        print('conv2 /shape:', x.shape)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        print('conv shape:', x.shape)
        x = self.conv3(x)
        print('conv3 shape:', x.shape)
        x = self.conv4(x)
        print('conv4 shape:', x.shape)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        print('conv shape:', x.shape)
        x = self.conv5(x)
        print('conv5 shape:', x.shape)
        x = x.view(-1, self.linearWidth)
        x = F.relu(self.fc1(x))
        return self.out(x)

    def output(self, w, f, p, s):
        result = (w - f + 2 * p) / s + 1
        return result


m = TrackNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)

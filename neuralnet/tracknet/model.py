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
        self.conv2 = BasicConv2d(in_ch=64, out_ch=256, k=3, s=2, p=1)
        self.conv3 = BasicConv2d(in_ch=256, out_ch=256, k=3, s=1, p=1)
        self.conv4 = BasicConv2d(in_ch=256, out_ch=128, k=3, s=1, p=1)
        self.conv5 = BasicConv2d(in_ch=128, out_ch=64, k=3, s=1, p=1)

        self.linearWidth = 64 * 2 * 2
        self.fc1 = nn.Linear(self.linearWidth, 64)
        self.out = nn.Linear(64, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=0)
        x = self.conv5(x)
        x = x.view(-1, self.linearWidth)
        x = F.relu(self.fc1(x))
        return self.out(x)


m = TrackNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print('Total Params:', torch_total_params)
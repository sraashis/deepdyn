def out_w(w, k, s, p):
    return ((w - k + 2 * p) / s) + 1


from torch import nn


class PatchNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(PatchNet, self).__init__()
        conv = [
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        self.conv = nn.Sequential(*conv)

        fc = [
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        ]

        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        x = self.conv.forward(x)
        x = x.view(64 * 4 * 4, -1)
        return self.fc.forward(x)


m = PatchNet(1, 2)
torch_total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(m)
print(out_w(31, 1, 1, 0))

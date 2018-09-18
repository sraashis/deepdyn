import torch.nn.functional as F
from torch import nn


class PatchNet(nn.Module):
    def __init__(self, width, channels, num_classes):
        super(PatchNet, self).__init__()

        self.channels = channels
        self.width = width

        self.kern_size = 5
        self.kern_stride = 2
        self.kern_padding = 1
        self.mxp_kern_size = 1
        self.mxp_stride = 1
        self.pool1 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv1 = nn.Conv2d(self.channels, 64, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()

        self.kern_size = 5
        self.kern_stride = 2
        self.kern_padding = 2
        self.mxp_kern_size = 1
        self.mxp_stride = 1
        self.pool2 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv2 = nn.Conv2d(64, 128, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()

        self.kern_size = 3
        self.kern_stride = 2
        self.kern_padding = 2
        self.mxp_kern_size = 1
        self.mxp_stride = 1
        self.pool3 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv3 = nn.Conv2d(128, 256, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()

        self.kern_size = 3
        self.kern_stride = 1
        self.kern_padding = 1
        self.mxp_kern_size = 2
        self.mxp_stride = 2
        self.pool4 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv4 = nn.Conv2d(256, 128, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()

        self.kern_size = 1
        self.kern_stride = 1
        self.kern_padding = 1
        self.mxp_kern_size = 1
        self.mxp_stride = 1
        self.pool5 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv5 = nn.Conv2d(128, 64, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()

        self.linearWidth = 64 * int(self.width) * int(self.width)
        self.fc1 = nn.Linear(self.linearWidth, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.dropout2d(x, p=0.3)
        x = self.pool4(F.relu(self.conv4(x)))
        x = F.dropout2d(x, p=0.3)
        x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(-1, self.linearWidth)
        x = F.dropout2d(x, p=0.4)
        x = F.relu(self.fc1(x))
        x = F.dropout2d(x, p=0.4)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def _update_output_size(self):
        temp = self.width
        self.width = ((self.width - self.kern_size + 2 * self.kern_padding) / self.kern_stride) + 1
        temp1 = self.width
        self.width = ((self.width - self.mxp_kern_size) / self.mxp_stride) + 1
        print('Output width[ ' + str(temp) + ' -conv-> ' + str(temp1) + ' -maxpool-> ' + str(self.width) + ' ]')

import numpy as np

i = PatchNet(32, 1, 1 )
model_parameters = filter(lambda p: p.requires_grad, i.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
# print(i)
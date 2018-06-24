import torch.nn.functional as F
from torch import nn


class PatchNet(nn.Module):
    def __init__(self, width, channels, num_classes):
        super(PatchNet, self).__init__()

        self.channels = channels
        self.width = width

        self.kern_size = 11
        self.kern_stride = 2
        self.kern_padding = 1
        self.mxp_kern_size = 1
        self.mxp_stride = 1
        self.pool1 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv1 = nn.Conv2d(self.channels, 128, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()

        self.kern_size = 5
        self.kern_stride = 1
        self.kern_padding = 3
        self.mxp_kern_size = 2
        self.mxp_stride = 2
        self.pool2 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv2 = nn.Conv2d(128, 256, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()

        self.kern_size = 5
        self.kern_stride = 1
        self.kern_padding = 1
        self.mxp_kern_size = 2
        self.mxp_stride = 2
        self.pool3 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv3 = nn.Conv2d(256, 512, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()

        self.kern_size = 3
        self.kern_stride = 2
        self.kern_padding = 1
        self.mxp_kern_size = 1
        self.mxp_stride = 1
        self.pool4 = nn.MaxPool2d(kernel_size=self.mxp_kern_size, stride=self.mxp_stride)
        self.conv4 = nn.Conv2d(512, 128, self.kern_size,
                               stride=self.kern_stride, padding=self.kern_padding)
        self._update_output_size()

        self.linearWidth = 128 * int(self.width) * int(self.width)
        self.fc1 = nn.Linear(self.linearWidth, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = x.view(-1, self.linearWidth)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

    def _update_output_size(self):
        temp = self.width
        self.width = ((self.width - self.kern_size + 2 * self.kern_padding) / self.kern_stride) + 1
        temp1 = self.width
        self.width = ((self.width - self.mxp_kern_size) / self.mxp_stride) + 1
        print('Output width[ ' + str(temp) + ' -conv-> ' + str(temp1) + ' -maxpool-> ' + str(self.width) + ' ]')

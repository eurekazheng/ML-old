from .BasicModule import BasicModule
import torch.nn as nn
import torchvision.models.alexnet


class DiaretNet(BasicModule):

    def __init__(self):
        super(DiaretNet, self).__init__()
        # Input 1@512
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 4@255
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 9, 4, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 9@126
        self.conv3 = nn.Sequential(
            nn.Conv2d(9, 16, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 16@62
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 25, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 25@30
        self.conv5 = nn.Sequential(
            nn.Conv2d(25, 36, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 36@14
        self.conv6 = nn.Sequential(
            nn.Conv2d(36, 49, 3, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 49@6
        self.conv7 = nn.Sequential(
            nn.Conv2d(49, 64, 3, 1, 0),
            nn.ReLU()
        )
        # 64@4
        self.dense = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        conv6_out = self.conv6(conv5_out)
        conv7_out = self.conv7(conv6_out)
        res = conv7_out.view(conv7_out.size(0), -1)
        out = self.dense(res)
        return out

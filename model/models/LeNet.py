# -*- coding: UTF-8 -*-
"""
@Project -> File   ：cnn_models_comparation.pytorch -> LeNet
@IDE    ：PyCharm
@Author ：QiangZiBro
@Date   ：2020/5/23 1:54 下午
@Desc   ：
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


# designed for cifar10
class LeNet(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # in_channels, out_channels, kernel_size, stride=1 ...
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2, 2))

        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

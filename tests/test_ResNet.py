# -*- coding: UTF-8 -*-
"""
@Project -> File   ：cnn_models_comparation.pytorch -> test_ResNet
@IDE    ：PyCharm
@Author ：QiangZiBro
@Date   ：2020/5/23 12:58 下午
@Desc   ：
"""
from unittest import TestCase
from model.models.ResNet import *
import torch


class TestResNet(TestCase):
    def test_for_resnet_any_input(self):
        model = resnet18(num_classes=10)
        img = torch.rand((32, 3, 32, 32))
        out = model(img)
        self.assertEqual((32,10), out.shape)

        img = torch.rand((32, 3, 320, 320))
        out = model(img)
        self.assertEqual((32,10), out.shape)

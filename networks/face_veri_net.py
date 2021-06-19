import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import scipy.io as sio
import os
import itertools
import shutil

# Internal libs
from core_dl.base_net import BaseNet
import networks.backbone_drn as drn


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class FaceVeriNet(BaseNet):

    def __init__(self, out_channel):
        super(FaceVeriNet, self).__init__()
        # drn_module = drn.drn_d_38(pretrained=True)  # use DRN38 for now
        # self.layer0 = nn.Sequential(
        #     nn.Conv2d(6, 16, kernel_size=7, stride=1, padding=3, bias=False),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True)
        # )
        #
        # self.block0 = nn.Sequential(
        #     drn_module.layer1
        # )
        # self.block1 = drn_module.layer2
        # self.block2 = drn_module.layer3
        # self.block3 = drn_module.layer4
        # self.block4 = drn_module.layer5
        # self.block5 = drn_module.layer6
        # self.block6 = nn.Sequential(
        #     drn_module.layer7,
        #     drn_module.layer8
        # )
        self.resnet = torchvision.models.resnet34(pretrained=True)
        self.output_layer = nn.Sequential(# nn.BatchNorm2d(512),
                                          # nn.Dropout(0.5),
                                          # nn.Flatten(),
                                          nn.Linear(512, out_channel),
                                          # nn.BatchNorm1d(out_channel)
                                         )
        # self.conv0 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

    def train(self, mode=True):
        super(FaceVeriNet, self).train(mode)

        def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        # for i in range(7):
        #     block = getattr(self, 'block' + str(i))
        #     block.apply(set_bn_eval)
        # self.resnet.apply(set_bn_eval)
        self.resnet.eval()

    def save_net_def(self, dir):
        super(FaceVeriNet, self).save_net_def(dir)
        shutil.copy(os.path.realpath(__file__), dir)

    def forward(self, input):
        """
        forward with image & scene feature
        :param image: (N, C, H, W)
        :return:
        """
        # x = self.layer0(x)
        # x0 = self.block0(x)  # 256
        # x1 = self.block1(x0)  # 128
        # x2 = self.block2(x1)  # 64
        # x3 = self.block3(x2)  # 32
        # x4 = self.block4(x3)  # 16
        # x5 = self.block5(x4)  # 8
        # x6 = self.block6(x5)  # 4
        # out = self.output_layer(x6)
        # return l2_norm(out)

        # feat = None
        # for i in range(2):
        #     x = input[:, :3, ...] if i == 0 else input[:, 3:, ...]
        #     # conv = getattr(self, 'conv' + str(i))
        #     # x = conv(x)
        #     x = self.resnet.conv1(x)
        #     x = self.resnet.bn1(x)
        #     x = self.resnet.relu(x)
        #     x = self.resnet.maxpool(x)
        #
        #     x = self.resnet.layer1(x)
        #     x = self.resnet.layer2(x)
        #     x = self.resnet.layer3(x)
        #     x = self.resnet.layer4(x)
        #
        #     x = self.resnet.avgpool(x)
        #     x = torch.flatten(x, 1)
        #
        #     feat = x if feat is None else torch.cat([feat, x], dim=1)

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        feat = torch.flatten(x, 1)

        out = self.output_layer(feat)
        return l2_norm(out)

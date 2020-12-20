# -*- coding:utf-8 -*-
# Author: moranli.aca@gmail.com
# Time: 2019/11/6 9:36
# FileName: Hourglass.py
# Descriptions: Here is a copy of the backbone used byLiuhao Ge.
# Paper: <3D Hand Shape and Pose Estimation from a Single RGB Image>
#

# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Networks for heatmap estimation from RGB images using Hourglass Network
"Stacked Hourglass Networks for Human Pose Estimation", Alejandro Newell, Kaiyu Yang, Jia Deng, ECCV 2016
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn


class _Residual_Module(nn.Module):
    def __init__(self, numIn, numOut):
        super(_Residual_Module, self).__init__()
        self.numIn, self.numOut = numIn, numOut
        self.bn = nn.BatchNorm2d(self.numIn)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.numOut // 2)
        self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual

class Hourglass_LiuhaoGe_CVPR2019(nn.Module):
    def __init__(self, n, nModules, nFeats):
        super(Hourglass_LiuhaoGe_CVPR2019, self).__init__()
        self.n, self.nModules, self.nFeats= n, nModules, nFeats

        _up1_, _low1_, _low2_, _low3_ = [], [], [], []
        for j in range(self.nModules):
            _up1_.append(_Residual_Module(self.nFeats, self.nFeats))
        self.low1 = nn.MaxPool2d(kernel_size=2, stride=2)
        for j in range(self.nModules):
            _low1_.append(_Residual_Module(self.nFeats, self.nFeats))

        if self.n > 1:
            self.low2 = Hourglass_LiuhaoGe_CVPR2019(n - 1, self.nModules, self.nFeats)
        else:
            for j in range(self.nModules):
                _low2_.append(_Residual_Module(self.nFeats, self.nFeats))
            self.low2_ = nn.ModuleList(_low2_)

        for j in range(self.nModules):
            _low3_.append(_Residual_Module(self.nFeats, self.nFeats))

        self.up1_ = nn.ModuleList(_up1_)
        self.low1_ = nn.ModuleList(_low1_)
        self.low3_ = nn.ModuleList(_low3_)
    def forward(self, x):
        up1 = x
        for j in range(self.nModules):
            up1 = self.up1_[j](up1)

        low1 = self.low1(x)
        for j in range(self.nModules):
            low1 = self.low1_[j](low1)

        if self.n > 1:
            low2 = self.low2(low1)
        else:
            low2 = low1
            for j in range(self.nModules):
                low2 = self.low2_[j](low2)

        low3 = low2
        for j in range(self.nModules):
            low3 = self.low3_[j](low3)
        # up2 = self.up2(low3)
        up2 = nn.functional.interpolate(low3,scale_factor=2)
        return up1 + up2


class Net_HG(nn.Module):
#     _Ge Liuhao_CVPR2019
    def __init__(self, num_joints, num_stages=2, num_modules=2, num_feats=256):
        super(Net_HG, self).__init__()

        self.numOutput, self.nStack, self.nModules, self.nFeats = num_joints, num_stages, num_modules, num_feats

        self.conv1_ = nn.Conv2d(3, 64, bias=True, kernel_size=7, stride=2, padding=3) ### 256 --> 128
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.r1 = _Residual_Module(64, 128)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.r4 = _Residual_Module(128, 128)
        self.r5 = _Residual_Module(128, self.nFeats)

        _hourglass, _Residual, _lin_, _tmpOut, _ll_, _tmpOut_, _reg_ = [], [], [], [], [], [], []
        for i in range(self.nStack):
            _hourglass.append(Hourglass_LiuhaoGe_CVPR2019(4, self.nModules, self.nFeats))
            for j in range(self.nModules):
                _Residual.append(_Residual_Module(self.nFeats, self.nFeats))
            lin = nn.Sequential(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1),
                                nn.BatchNorm2d(self.nFeats), self.relu)
            _lin_.append(lin)
            _tmpOut.append(nn.Conv2d(self.nFeats, self.numOutput, bias=True, kernel_size=1, stride=1))
            if i < self.nStack - 1:
                _ll_.append(nn.Conv2d(self.nFeats, self.nFeats, bias=True, kernel_size=1, stride=1))
                _tmpOut_.append(nn.Conv2d(self.numOutput, self.nFeats, bias=True, kernel_size=1, stride=1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.lin_ = nn.ModuleList(_lin_)
        self.tmpOut = nn.ModuleList(_tmpOut)
        self.ll_ = nn.ModuleList(_ll_)
        self.tmpOut_ = nn.ModuleList(_tmpOut_)

    def forward(self, x):
        x = self.conv1_(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.r1(x)
        x = self.maxpool(x)
        x = self.r4(x)
        x = self.r5(x)

        out = []
        encoding = []
        for i in range(self.nStack):
            hg = self.hourglass[i](x) ##Bz, 256, 64, 64
            ll = hg
            for j in range(self.nModules):
                ll = self.Residual[i * self.nModules + j](ll)
            ll = self.lin_[i](ll)
            tmpOut = self.tmpOut[i](ll)

            out.append(tmpOut)
            if i < self.nStack - 1:
                ll_ = self.ll_[i](ll)
                tmpOut_ = self.tmpOut_[i](tmpOut)
                x = x + ll_ + tmpOut_
                encoding.append(x)
            else:
                encoding.append(ll)

        return out, encoding



# if __name__ == '__main__':
#     import torch
#     import torch.nn as nn
#     from torch.autograd import Variable
#
#     Bz, knum, size = 8, 21, 128
#     HyperK = 3
#     nG, Gk = 2, 4
#     # nG, Gk = 2, 4
#     input = Variable(torch.rand(Bz, 3, size, size), requires_grad=True)
#     planes = [32, 32, 64, 128, 128, 256, 256]
#     Config = [nG, Gk]
#     net = Net_Latent25D_ECCV2018(knum)
#     # print(net.depthwise_conv1[0].weight)
#     out = net(input)
#     # print(out.shape)
#     # print(encoding)
#     # print(out.shape, encoding.shape)
#     # p, t, o = xyz_pred1
#     # print(p, t, o)
#     # check = torch.autograd.gradcheck(net, input, eps=1e-06, atol=1e-04, raise_exception=True)
#     # print(check)
#     def print_model_param_num(model):
#         total = sum([param.nelement() for param in model.parameters()])
#         print("==> Model Params: %2.3fM" % (total / 1e6))
#     print_model_param_num(net)
#     # print(xyz_pred1.shape, 'xxx')
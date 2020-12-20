# -*- coding:utf-8 -*-
# Author: moranli.aca@gmail.com
# Time: 2019/10/9 17:02
# FileName: Block_lib.py
# Descriptions: some general block
# DownsampleConv2D_R: a better version of downsample


from __future__ import absolute_import, division
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class small_HG(nn.Module):
    def __init__(self, planes):
        super(small_HG, self).__init__()
        self.conv1 = Res2DBlock(planes[0], planes[1])
        self.down1 = DownsampleConv2D_R(planes[1], planes[1])
        self.down2 = DownsampleConv2D_R(planes[1], planes[1])
        self.up2 = Upsample2DBlock_R(planes[1], planes[1], 4, 2)
        self.up1 = nn.Sequential(Basic2DBlock(planes[1] + planes[1], planes[1], 1),
                                 Upsample2DBlock_R(planes[1], planes[1], 4, 2))
        self.up0 = nn.Sequential(Basic2DBlock(planes[0] + 2 * planes[1], 2 * planes[1], 1),
                                 Res2DBlock(2 * planes[1], planes[1]))

    def forward(self, x):
        l0 = self.conv1(x)
        l1 = self.down1(l0)
        l2 = self.down2(l1)
        up2 = self.up2(l2)
        up1 = self.up1(torch.cat([up2, l1], 1))
        up0 = self.up0(torch.cat([x, l0, up1], 1))
        return up0

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=1, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

        assert self.inter_channels > 0

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            bn = nn.BatchNorm2d
        elif dimension == 1:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d
        else:
            raise Exception('Error feature dimension.')

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.concat_project = nn.Sequential(
            nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
            nn.ReLU()
        )

        nn.init.kaiming_normal_(self.concat_project[0].weight)
        nn.init.kaiming_normal_(self.g.weight)
        nn.init.constant_(self.g.bias, 0)
        nn.init.kaiming_normal_(self.theta.weight)
        nn.init.constant_(self.theta.bias, 0)
        nn.init.kaiming_normal_(self.phi.weight)
        nn.init.constant_(self.phi.bias, 0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.kaiming_normal_(self.W[0].weight)
            nn.init.constant_(self.W[0].bias, 0)
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        if sub_sample > 1:

            self.g = nn.Sequential(self.g, max_pool(kernel_size=sub_sample))
            self.phi = nn.Sequential(self.phi, max_pool(kernel_size=sub_sample))


    def forward(self, x):
        batch_size = x.size(0)  # x: (b, c, h, w)
        ## N1 = (h/subsample)*(w/subsample)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1) ## Bz,interC,(h/subsample)*(w/subsample)

        g_x = g_x.permute(0, 2, 1) ## Bz, N1, interC
        ## N=h*w
        # (b, c, N, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)


        # (b, c, 1, N1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        h = theta_x.size(2) ## N
        w = phi_x.size(3) ## N1
        theta_x = theta_x.expand(-1, -1, -1, w) ## Bz, C, N, N1
        phi_x = phi_x.expand(-1, -1, h, -1) ## Bz, C, N, N1
        concat_feature = torch.cat([theta_x, phi_x], dim=1) ## Bz, 2C, N,N1

        f = self.concat_project(concat_feature) ##Bz,1,N,N1
        b, _, h, w = f.size()
        f = f.view(b, h, w) ##Bz,N,N1

        N = f.size(-1) ##N1
        f_div_C = f / N
        y = torch.matmul(f_div_C, g_x) ##Bz, N, interC
        y = y.permute(0, 2, 1).contiguous() ## Bz, interC, N
        y = y.view(batch_size, self.inter_channels, *x.size()[2:]) ## Bz, interC,h,w
        W_y = self.W(y)
        z = W_y + x ## Bz, interC,h,w

        return z

class Basic1DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic1DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size - 1) // 2)),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Res1DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res1DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(True),
            nn.Conv1d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)





class Res1DBlock_K(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Res1DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(True),
            nn.Conv1d(out_planes, out_planes, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm1d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)







class DownsampleConv(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownsampleConv, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(True),
            nn.Conv1d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        res = self.res_branch(x)

        return res










class Upsample1DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample1DBlock, self).__init__()
        assert (kernel_size == 2)
        assert (stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0,
                               output_padding=0),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)




class Upsample1DBlock_R(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample1DBlock_R, self).__init__()
        assert (kernel_size == 4)
        assert (stride == 2)
        self.block = nn.Sequential(
            # nn.ConvTranspose2d(3, 4, kernel_size=4, padding=1, stride=2)
            nn.ConvTranspose1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm1d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Basic2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic2DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size - 1) // 2)),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Res2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res2DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)







class Pool2DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool2DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)

class Upsample2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample2DBlock, self).__init__()
        assert (kernel_size == 2)
        assert (stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0,
                               output_padding=0),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class Upsample2DBlock_R(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample2DBlock_R, self).__init__()
        assert (kernel_size == 4)
        assert (stride == 2)
        self.block = nn.Sequential(
            # nn.ConvTranspose2d(3, 4, kernel_size=4, padding=1, stride=2)
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)







class DownsampleConv2D(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownsampleConv2D, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        res = self.res_branch(x)

        return res


class DownsampleConv2D_R(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownsampleConv2D_R, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )
        # if in_planes == out_planes:
        #     self.skip_con = nn.Sequential()
        # else:
        self.skip_con = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=2),
                                          nn.BatchNorm2d(out_planes))


    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res+skip, True)



class SoftArgMax(nn.Module):
    def __init__(self, size): ## for Insize = Bz, 21, k
        super(SoftArgMax, self).__init__()
        self.knum, self.Group_k = size
        self.beta = nn.Conv1d(self.knum, self.knum, 1, 1, 0, groups=self.knum, bias=False)
        self.wx = torch.arange(0.0, 1.0*self.knum, 1).view(self.knum, 1).repeat(1, self.Group_k)
        self.wx = nn.Parameter(self.wx, requires_grad=False)
        nn.init.constant_(self.beta.weight, 1000)
        # self.beta.weight = nn.Parameter(torch.tensor(1000.0))
    def forward(self, x):
        scoremap = self.beta(x)
        scoremap = F.softmax(scoremap,  dim=1)
        scoremap = scoremap.mul(self.wx)
        arg_max_res = torch.sum(scoremap, dim=1)
        return arg_max_res

class SoftHeatmap(nn.Module):
    def __init__(self, size, kp_num):
        super(SoftHeatmap, self).__init__()
        self.size = size
        self.beta = nn.Conv2d(kp_num, kp_num, 1, 1, 0, groups=kp_num, bias=False)
        self.wx = torch.arange(0.0, 1.0 * self.size, 1).view([1, self.size]).repeat([self.size, 1])
        self.wy = torch.arange(0.0, 1.0 * self.size, 1).view([self.size, 1]).repeat([1, self.size])
        self.wx = nn.Parameter(self.wx, requires_grad=False)
        self.wy = nn.Parameter(self.wy, requires_grad=False)
    def forward(self, x):
        s = list(x.size())
        scoremap = self.beta(x)
        scoremap = scoremap.view([s[0], s[1], s[2] * s[3]])
        scoremap = F.softmax(scoremap, dim=2)
        scoremap = scoremap.view([s[0], s[1], s[2], s[3]])
        scoremap_x = scoremap.mul(self.wx)
        scoremap_x = scoremap_x.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_x = torch.sum(scoremap_x, dim=2)
        scoremap_y = scoremap.mul(self.wy)
        scoremap_y = scoremap_y.view([s[0], s[1], s[2] * s[3]])
        soft_argmax_y = torch.sum(scoremap_y, dim=2)
        keypoint_uv = torch.stack([soft_argmax_x, soft_argmax_y], dim=2)

        return keypoint_uv, scoremap

class nddr_layer(nn.Module):
    def __init__(self, in_channels, out_channels, task_num, init_weights=[0.9, 0.1], init_method='constant'):
        super(nddr_layer, self).__init__()
        self.task_num = task_num
        assert task_num>=2, 'Task Num Must >=2'


        self.Conv_Task_List = nn.ModuleList([])
        for i in range(self.task_num):
            task_basic = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU(True))
            self.Conv_Task_List.append(task_basic)

        self_w,  others_w = init_weights[0],  init_weights[1]/(self.task_num-1)
        others_diag = others_w * np.diag(np.ones(out_channels)).astype(dtype=np.float32)
        self_w_diag = self_w * np.diag(np.ones(out_channels))

        if init_method == 'constant':
            for i in range(self.task_num):
                diag_M = np.tile(others_diag, (1, self.task_num))
                start_id = int(i*out_channels)
                end_id = start_id + out_channels
                diag_M[:, start_id:end_id] = self_w_diag
                self.Conv_Task_List[i][0].weight = torch.nn.Parameter(torch.from_numpy(diag_M[:,:,np.newaxis, np.newaxis]))
                torch.nn.init.constant_(self.Conv_Task_List[i][0].bias, 0.0)
    def forward(self, Net_F):
        Net_Res = []
        for i in range(self.task_num):
            tmp = self.Conv_Task_List[i](Net_F)
            Net_Res.append(tmp)

        return Net_Res




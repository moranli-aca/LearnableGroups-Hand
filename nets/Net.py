# -*- coding:utf-8 -*-
# Author: moranli.aca@gmail.com
# Time: 2020/12/20 2:04 PM
# FileName: Net.py
# Descriptions: 
#


import torch
import torch.nn as nn
from nets.Backbone import Net_HG
from nets.Block_lib import SoftHeatmap, Res2DBlock, Basic2DBlock, DownsampleConv2D_R, Upsample2DBlock_R, nddr_layer


class Learnable_Groups(nn.Module):
    def __init__(self, Config, in_size, kp_num, planes):
        '''
        :param Config: [number of Groups, nddr_flag]
        :param in_size, kp_num, planes: #channels for sub-branches
        '''
        super(Learnable_Groups, self).__init__()

        self.nG, self.nddr = Config[0], Config[1]
        self.in_size, self.kp_num = in_size, kp_num
        self.backbone = Net_HG(self.kp_num, num_stages=1) 
        self.backbone_conv = nn.Sequential(Res2DBlock(256 + self.kp_num, 128),
                                           Res2DBlock(128, planes[0]))
        # self.get_uv0 = SoftHeatmap(in_size // 4, self.kp_num)
        _b_conv0, _b_htmaps_conv, _b_get_uv, _b_depth_conv = [], [], [], []
        _b_conv1, _b_down1, _b_down2, _b_up1, _b_up2, _b_up0 = [], [], [], [], [], []
        self.meta_S = nn.Linear(self.nG, self.kp_num, bias=False)
        nn.init.constant_(self.meta_S.weight, 0.5)

        for j in range(self.nG):
            _b_conv0.append(Basic2DBlock(planes[0], planes[0], 1))
            _b_conv1.append(Res2DBlock(planes[0], planes[1]))
            _b_down1.append(DownsampleConv2D_R(planes[1], planes[1]))
            _b_down2.append(DownsampleConv2D_R(planes[1], planes[1]))
            _b_up1.append(nn.Sequential(Basic2DBlock(2 * planes[1], planes[1], 1),
                                        Upsample2DBlock_R(planes[1], planes[1], 4, 2)))
            _b_up2.append(nn.Sequential(Upsample2DBlock_R(planes[1], planes[1], 4, 2)))
            _b_up0.append(nn.Sequential(Basic2DBlock(planes[0] + 2 * planes[1], 2 * planes[1], 1),
                                        Res2DBlock(2 * planes[1], planes[1])))

            _b_htmaps_conv.append(nn.Sequential(nn.Conv2d(planes[1], planes[0], 3, 1, 1),
                                                nn.ReLU(True),
                                                nn.Conv2d(planes[0], self.kp_num, 3, 1, 1)))
            _b_get_uv.append(SoftHeatmap(self.in_size // 4, self.kp_num))
            _b_depth_conv.append(nn.Sequential(nn.Conv2d(planes[1], planes[0], 3, 1, 1),
                                               nn.ReLU(True),
                                               nn.Conv2d(planes[0], self.kp_num, 3, 1, 1)))

        self.b_conv0, self.b_conv1= nn.ModuleList(_b_conv0),  nn.ModuleList(_b_conv1)
        self.b_down1, self.b_down2 = nn.ModuleList(_b_down1), nn.ModuleList(_b_down2)
        self.b_up2, self.b_up1 = nn.ModuleList(_b_up2), nn.ModuleList(_b_up1)
        self.b_up0 = nn.ModuleList(_b_up0)
        self.b_htmaps_conv = nn.ModuleList(_b_htmaps_conv)
        self.b_get_uv, self.b_depth_conv = nn.ModuleList(_b_get_uv), nn.ModuleList(_b_depth_conv)

        if self.nddr:
            print('>> Adding nddr layer...')
            self.nddr_layer0 = nddr_layer(self.nG * planes[1], planes[1], self.nG)

    def forward(self, x, temperature):
        _htmap0, _T0 = self.backbone(x)
        htmap0, T0 = _htmap0[0], _T0[0]
        # weak_kpt_uv, __ = self.get_uv0(htmap0)
        kpt_F1 = self.backbone_conv(torch.cat([htmap0, T0], 1))
        S_list = list(kpt_F1.shape)

        ## Bz, knum, P0,in/4, in/4
        _b_depth, _b_uv, _BF_1 = [], [], []

        beta2 = torch.tensor(temperature)
        m = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(beta2, logits=self.meta_S.weight)
        meta_S2 = m.rsample()
        for j in range(self.nG):
            bf_0 = self.b_conv0[j](kpt_F1)
            bf_1 = self.b_conv1[j](bf_0)
            bf_2 = self.b_down1[j](bf_1)
            bf_3 = self.b_down2[j](bf_2)
            bf_31 = self.b_up2[j](bf_3)
            bf_21 = self.b_up1[j](torch.cat([bf_31, bf_2], 1))
            BF_1 = self.b_up0[j](torch.cat([bf_21, bf_1, bf_0], 1))
            _BF_1.append(BF_1)

        BF_1_cat = self.nddr_layer0(torch.cat(_BF_1, 1)) if self.nddr else _BF_1
        for j in range(self.nG):
            b_htmaps = self.b_htmaps_conv[j](BF_1_cat[j])  ## Bz, 21, in/4, in/4
            b_depthmaps = self.b_depth_conv[j](BF_1_cat[j])  # Bz, 21, in/4, in/4
            b_uv, b_scoremap = self.b_get_uv[j](b_htmaps)  ##Bz, 21, 2
            b_depth = b_depthmaps.mul(b_scoremap)
            b_depth = b_depth.view(S_list[0], self.kp_num, S_list[-2] * S_list[-1])
            b_depth = torch.sum(b_depth, dim=2)  ##Bz, 21
            b_uv = b_uv * meta_S2[:, j].view(1, -1, 1).expand_as(b_uv)
            b_depth = b_depth * meta_S2[:, j].expand_as(b_depth)
            _b_depth.append(b_depth.unsqueeze(-1))
            _b_uv.append(b_uv.unsqueeze(-1))

        depth_res = torch.sum(torch.cat(_b_depth, -1), dim=-1)
        uv_res = torch.sum(torch.cat(_b_uv, -1), dim=-1)
        return 4.0 * uv_res, depth_res





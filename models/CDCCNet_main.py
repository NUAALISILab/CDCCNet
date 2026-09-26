import math
from re import L
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function

import numpy as np
import random
import matplotlib.pyplot as plt
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math
import sys
sys.path.append('./')
from models import Resnet
from models.help_funcs import TwoLayerConv2d
from torch.backends import cudnn
import logging
from models.instance_whitening import compute_covariance_matrix, cross_covariance_diagonal_loss
from models.mynn import initialize_weights, Norm2d, Upsample, freeze_weights, unfreeze_weights
from models.STCCL import STCCLLoss, ProjectionHead
from models.PCRCL import PCRCLLoss


def _mark_skip_init(module):
    setattr(module, 'skip_init', True)
    for child in module.children():
        _mark_skip_init(child)

class Conv3Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1),
                                               stride=(stride, stride), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x
class _AtrousSpatialPyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn
        print("output_stride = ", output_stride)
        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 4:
            rates = [4 * r for r in rates]
        elif output_stride == 16:
            pass
        elif output_stride == 32:
            rates = [r // 2 for r in rates]
        else:
            raise ValueError('output stride of {} not supported'.format(output_stride))

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            Norm2d(256), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class Model_Architecture(nn.Module):
    def __init__(self, num_classes=2, backbone='resnet-50', wt_layer=[0, 0, 1, 1, 1, 0, 0], output_sigmoid=False,
                 fusion_policy='concat', variant='D16', args=None):
        super(Model_Architecture, self).__init__()
        self.wt_layer = wt_layer
        self.backbone = backbone
        self.variant = variant
        self.args = args
        self.neck_layer = FeatureFusion(fusion_policy)
        self.output_sigmoid = output_sigmoid
        self.num_classes = num_classes
        self.cls_head = TwoLayerConv2d(in_channels=2*2 if fusion_policy == "concat" else 2, out_channels=num_classes)
        self.cls_head1 = TwoLayerConv2d(in_channels=num_classes, out_channels=num_classes)
        if self.training:
            self.criterion_CA = nn.MSELoss()
            self.criterion_STCCL = STCCLLoss(self.args)
            self.criterion_PCRCL = PCRCLLoss(self.args)

        if self.backbone == 'resnet-18':
            channel_1st = 3
            channel_2nd = 64
            channel_3rd = 64
            channel_4th = 128
            prev_final_channel = 256
            final_channel = 512
            resnet = Resnet.resnet18(wt_layer=self.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif backbone == 'resnet-50':
            channel_1st = 3
            channel_2nd = 64
            channel_3rd = 256
            channel_4th = 512
            prev_final_channel = 1024
            final_channel = 2048
            resnet = Resnet.resnet50(wt_layer=self.wt_layer)
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D16':
            os = 16
            for n, m in self.layer4.named_modules():
                if 'conv1' in n:
                    m.stride = (1, 1)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            # raise 'unknown deepv3 variant: {}'.format(self.variant)
            print("Not using Dilation ")

        if self.variant == 'D':
            os = 8
        elif self.variant == 'D4':
            os = 4
        elif self.variant == 'D16':
            os = 16
            self.stage1_Conv1 = Conv3Relu(final_channel * 2,
                                          final_channel)  # channel: 2*final_channel ---> final_channel
            self.stage1_Conv2 = Conv3Relu(512, 256)
            self.stage1_Conv3 = Conv3Relu(4096, 2048)
        else:
            os = 32
            self.stage1_Conv1 = Conv3Relu(final_channel * 2,
                                          final_channel)  # channel: 2*final_channel ---> final_channel
            self.stage1_Conv2 = Conv3Relu(128, 64)
            self.stage1_Conv3 = Conv3Relu(1024, 512)



        self.output_stride = os
        self.aspp = _AtrousSpatialPyramidPoolingModule(final_channel, 256,
                                                       output_stride=os)

        self.bot_fine = nn.Sequential(
            nn.Conv2d(channel_3rd, 48, kernel_size=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True))

        self.bot_aspp = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final1 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))

        self.final2 = nn.Sequential(
            nn.Conv2d(256, self.num_classes, kernel_size=1, bias=True))

        self.dsn = nn.Sequential(
            nn.Conv2d(prev_final_channel, 512, kernel_size=3, stride=1, padding=1),
            Norm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        initialize_weights(self.dsn)
        _mark_skip_init(self.dsn)
        initialize_weights(self.aspp)
        _mark_skip_init(self.aspp)
        initialize_weights(self.bot_aspp)
        _mark_skip_init(self.bot_aspp)
        initialize_weights(self.bot_fine)
        _mark_skip_init(self.bot_fine)
        initialize_weights(self.final1)
        _mark_skip_init(self.final1)
        initialize_weights(self.final2)
        _mark_skip_init(self.final2)

        # Setting the flags
        self.eps = 1e-5
        self.whitening = False

        if self.backbone == 'resnet-18':
            self.three_input_layer = False
            in_channel_list = [0, 0, 64, 64, 128, 256, 512]
            out_channel_list = [0, 0, 32, 32, 64,  128, 256]
        else: # ResNet-50
            self.three_input_layer = False
            in_channel_list = [0, 0, 64, 256, 512, 1024, 2048]
            out_channel_list = [0, 0, 32, 128, 256,  512, 1024]


        # Projection head for Contrastive Learning
        self.ProjectionHead_cls_0 = ProjectionHead(dim_in=in_channel_list[-1])
        self.ProjectionHead_cls_1 = ProjectionHead(dim_in=304)
        self.ProjectionHead_cls_2 = ProjectionHead(dim_in=256)

        initialize_weights(self.ProjectionHead_cls_0)
        _mark_skip_init(self.ProjectionHead_cls_0)
        initialize_weights(self.ProjectionHead_cls_1)
        _mark_skip_init(self.ProjectionHead_cls_1)
        initialize_weights(self.ProjectionHead_cls_2)
        _mark_skip_init(self.ProjectionHead_cls_2)

    def forward(self, x1, x2, x1_c=None, x2_c=None, gt=None):
        if self.training:
            out, out_c, aux_out, ACML1, ACML2, CCDL1, CCDL2, STCCL, PCRCL = self.forward_encoder_decoder(x1, x2, x1_c, x2_c, gt)
            return out, out_c, aux_out, ACML1, ACML2, CCDL1, CCDL2, STCCL, PCRCL
        else:
            out = self.forward_encoder(x1, x2)
            return out

    def forward_encoder_decoder(self, x1, x2, x1_c=None, x2_c=None, gt=None):
        k_arr1 = []
        q_arr1 = []
        k_arr2 = []
        q_arr2 = []

        kd_arr = []
        qd_arr = []
        x_size = x1.size()

        x1 = self.layer0[0](x1)
        x2 = self.layer0[0](x2)
        if self.wt_layer[2] == 1 or self.wt_layer[2] == 2:
            x1, k1 = self.layer0[1](x1)
            k_arr1.append(k1)
            x2, k2 = self.layer0[1](x2)
            k_arr2.append(k2)
        else:
            x1 = self.layer0[1](x1)
            x2 = self.layer0[1](x2)
        x1 = self.layer0[2](x1)
        x1 = self.layer0[3](x1)
        x2 = self.layer0[2](x2)
        x2 = self.layer0[3](x2)

        if self.training:
            x1_c = self.layer0[0](x1_c)
            x1_c, q1 = self.layer0[1](x1_c)
            q_arr1.append(q1)
            x1_c = self.layer0[2](x1_c)
            x1_c = self.layer0[3](x1_c)

            x2_c = self.layer0[0](x2_c)
            x2_c, q2 = self.layer0[1](x2_c)
            q_arr2.append(q2)
            x2_c = self.layer0[2](x2_c)
            x2_c = self.layer0[3](x2_c)

        x_tuple1 = self.layer1([x1, k_arr1])
        low_level1 = x_tuple1[0]
        x_tuple1 = self.layer2(x_tuple1)
        x_tuple1 = self.layer3(x_tuple1)

        aux_out1 = x_tuple1[0]
        x_tuple1 = self.layer4(x_tuple1)
        x1 = x_tuple1[0]
        k_arr1 = x_tuple1[1]
        kd_arr1 = x_tuple1[0]

        if self.training:
            x1_c_tuple = self.layer1([x1_c, q_arr1])
            low_level1_c = x1_c_tuple[0]

            x1_c_tuple = self.layer2(x1_c_tuple)
            x1_c_tuple = self.layer3(x1_c_tuple)

            x1_c_tuple = self.layer4(x1_c_tuple)
            x1_c = x1_c_tuple[0]
            q_arr1 = x1_c_tuple[1]
            qd_arr1 = x1_c_tuple[0]

        x_tuple2 = self.layer1([x2, k_arr2])
        low_level2 = x_tuple2[0]
        x_tuple2 = self.layer2(x_tuple2)
        x_tuple2 = self.layer3(x_tuple2)
        aux_out2 = x_tuple2[0]
        x_tuple2 = self.layer4(x_tuple2)
        x2 = x_tuple2[0]
        k_arr2 = x_tuple2[1]
        kd_arr2 = x_tuple2[0]

        if self.training:
            kd_arr12 = self.stage1_Conv3(torch.cat([kd_arr1, kd_arr2], 1))
            kd_arr.append(kd_arr12)

        x = self.stage1_Conv1(torch.cat([x1, x2], 1))
        low_level = self.stage1_Conv2(torch.cat([low_level1, low_level2], 1))
        x = self.aspp(x)
        dec0_up = self.bot_aspp(x)
        dec0_fine = self.bot_fine(low_level)
        dec0_up = Upsample(dec0_up, low_level.size()[2:])
        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)

        if self.training:
            kd_arr.append(dec0)
        dec1 = self.final1(dec0)

        if self.training:
            kd_arr.append(dec1)
        dec2 = self.final2(dec1)

        main_out = Upsample(dec2, x_size[2:])
        out = self.cls_head1(main_out)

        if self.training:
            x2_c_tuple = self.layer1([x2_c, q_arr2])
            low_level2_c = x2_c_tuple[0]
            x2_c_tuple = self.layer2(x2_c_tuple)
            x2_c_tuple = self.layer3(x2_c_tuple)
            x2_c_tuple = self.layer4(x2_c_tuple)
            x2_c = x2_c_tuple[0]
            q_arr2 = x2_c_tuple[1]
            qd_arr2 = x2_c_tuple[0]
            qd_arr12 = self.stage1_Conv3(torch.cat([qd_arr1, qd_arr2], 1))
            qd_arr.append(qd_arr12)

            x_c = self.stage1_Conv1(torch.cat([x1_c, x2_c], 1))
            low_level_c = self.stage1_Conv2(torch.cat([low_level1_c, low_level2_c], 1))
            x_c = self.aspp(x_c)
            dec0_up_c = self.bot_aspp(x_c)
            dec0_fine_c = self.bot_fine(low_level_c)
            dec0_up_c = Upsample(dec0_up_c, low_level_c.size()[2:])
            dec0_c = [dec0_fine_c, dec0_up_c]
            dec0_c = torch.cat(dec0_c, 1)
            qd_arr.append(dec0_c)
            dec1_c = self.final1(dec0_c)
            qd_arr.append(dec1_c)
            dec2_c = self.final2(dec1_c)
            main_out_c = Upsample(dec2_c, x_size[2:])
            out_c = self.cls_head1(main_out_c)

            aux_out1 = self.dsn(aux_out1)
            aux_out2 = self.dsn(aux_out2)
            x = self.neck_layer.fusion(aux_out1, aux_out2, self.neck_layer.policy)
            aux_out = self.cls_head(x)


            _, predict = torch.max(dec2, 1)
            _, predict_c = torch.max(dec2_c, 1)

            ACML1 = torch.FloatTensor([0]).cuda()
            CCDL1 = torch.FloatTensor([0]).cuda()

            for N, f_maps in enumerate(zip(k_arr1, q_arr1)):
                k1_maps, q1_maps = f_maps
                k1_cor, _ = compute_covariance_matrix(k1_maps)
                q1_cor, _ = compute_covariance_matrix(q1_maps)
                cov_loss1 = self.criterion_CA(k1_cor, q1_cor)
                crosscov_loss1 = cross_covariance_diagonal_loss(k1_maps, q1_maps)
                ACML1 = ACML1 + cov_loss1
                CCDL1 = CCDL1 + crosscov_loss1
            ACML1 = ACML1 / len(k_arr1)
            CCDL1 = CCDL1 / len(k_arr1)
            ACML2 = torch.FloatTensor([0]).cuda()
            CCDL2 = torch.FloatTensor([0]).cuda()

            for N, f_maps in enumerate(zip(k_arr2, q_arr2)):
                k2_maps, q2_maps = f_maps
                k2_cor, _ = compute_covariance_matrix(k2_maps)
                q2_cor, _ = compute_covariance_matrix(q2_maps)
                cov_loss2 = self.criterion_CA(k2_cor, q2_cor)
                crosscov_loss2 = cross_covariance_diagonal_loss(k2_maps, q2_maps)
                ACML2 = ACML2 + cov_loss2
                CCDL2 = CCDL2 + crosscov_loss2
            ACML2 = ACML2 / len(k_arr2)
            CCDL2 = CCDL2 / len(k_arr2)

            STCCL = torch.FloatTensor([0]).cuda()
            PCRCL = torch.FloatTensor([0]).cuda()
            for N, f_maps in enumerate(zip(qd_arr, kd_arr)):
                feat_q, feat_k = f_maps
                projection = getattr(self, 'ProjectionHead_cls_%d' % N)
                embed_q = projection(feat_q)
                embed_k = projection(feat_k)
                loss_STCCL = self.criterion_STCCL(embed_q, embed_k, gt, predict)
                STCCL = STCCL + loss_STCCL.mean()
                loss_PCRCL = self.criterion_PCRCL(embed_q, embed_k, predict, predict_c, gt)
                PCRCL = PCRCL + loss_PCRCL.mean()
            STCCL = STCCL / len(qd_arr)
            PCRCL = PCRCL / len(qd_arr)

            return out, out_c, aux_out, ACML1, ACML2, CCDL1, CCDL2, STCCL, PCRCL

        else:
            return out

class FeatureFusion(nn.Module):
    def __init__(self,
                 policy,
                 in_channels=None,
                 channels=None,
                 out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.policy = policy
        self.in_channels = in_channels
        self.channels = channels
        self.out_indices = out_indices

    @staticmethod
    def fusion(x1, x2, policy):
        """Specify the form of feature fusion"""

        _fusion_policies = ['concat', 'sum', 'diff', 'abs_diff']
        assert policy in _fusion_policies, 'The fusion policies {} are ' \
                                           'supported'.format(_fusion_policies)

        if policy == 'concat':
            x = torch.cat([x1, x2], dim=1)
        elif policy == 'sum':
            x = x1 + x2
        elif policy == 'diff':
            x = x2 - x1
        elif policy == 'abs_diff':
            x = torch.abs(x1 - x2)

        return x

    def forward(self, x1, x2):
        """Forward function."""

        assert len(x1) == len(x2), "The features x1 and x2 from the" \
                                   "backbone should be of equal length"
        outs = []
        for i in range(len(x1)):
            out = self.fusion(x1[i], x2[i], self.policy)
            outs.append(out)

        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)

class CDCCNet(nn.Module):
    def __init__(self, num_classes=2, backbone='resnet-50', wt_layer=[0, 0, 1, 1, 1, 0, 0], output_sigmoid=False, fusion_policy='concat',
                 variant='D16', args = None):
        super(CDCCNet, self).__init__()
        self.encoder_decoder = Model_Architecture(num_classes, backbone, wt_layer, output_sigmoid, fusion_policy, variant, args)

    def forward(self, x1, x2, x1_color=None, x2_color=None, gt=None):
        if self.training == True:
            out, out_c, aux_out, ACML1, ACML2, CCDL1, CCDL2, STCCL, PCRCL = self.encoder_decoder(x1, x2, x1_color, x2_color, gt)
            return out, out_c, aux_out, ACML1, ACML2, CCDL1, CCDL2, STCCL, PCRCL
        else:
            out = self.encoder(x1, x2)
            return out
import torch
from torch import nn
import sys
sys.path.append('./')
from models import Resnet
from models.help_funcs import TwoLayerConv2d
from models.mynn import initialize_weights, Norm2d, Upsample
from models.STCCL import STCCLLoss
from models.PCRCL import PCRCLLoss
from models.covariance_matrix import compute_covariance_matrix, cross_covariance_diagonal_loss


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        )
        self.l2norm = Normalize(2)

    def forward(self, x):
        return self.l2norm(self.proj(x))


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

    def _mean_feature_constraint_losses(self, original_features, augmented_features):
        acm_total = torch.FloatTensor([0]).cuda()
        ccd_total = torch.FloatTensor([0]).cuda()

        for original_map, augmented_map in zip(original_features, augmented_features):
            original_covariance, _ = compute_covariance_matrix(original_map)
            augmented_covariance, _ = compute_covariance_matrix(augmented_map)

            acm_total = acm_total + self.criterion_CA(original_covariance, augmented_covariance)
            ccd_total = ccd_total + cross_covariance_diagonal_loss(original_map, augmented_map)

        feature_level_count = len(original_features)
        return acm_total / feature_level_count, ccd_total / feature_level_count

    def _mean_decoder_contrastive_losses(self, augmented_features, original_features, gt, original_prediction, augmented_prediction):
        stccl_total = torch.FloatTensor([0]).cuda()
        pcrcl_total = torch.FloatTensor([0]).cuda()

        for level_index, (augmented_map, original_map) in enumerate(zip(augmented_features, original_features)):
            projection_head = getattr(self, 'ProjectionHead_cls_%d' % level_index)
            augmented_embedding = projection_head(augmented_map)
            original_embedding = projection_head(original_map)

            stccl_level = self.criterion_STCCL(augmented_embedding, original_embedding, gt, original_prediction)
            stccl_total = stccl_total + stccl_level.mean()

            pcrcl_level = self.criterion_PCRCL(augmented_embedding, original_embedding, original_prediction, augmented_prediction, gt)
            pcrcl_total = pcrcl_total + pcrcl_level.mean()

        decoder_level_count = len(augmented_features)
        return stccl_total / decoder_level_count, pcrcl_total / decoder_level_count


    def forward(self, x1_ori, x2_ori, x1_aug=None, x2_aug=None, gt=None):
        if self.training:
            out_ori, out_aug, aux_out, ACML1, ACML2, CCDL1, CCDL2, STCCL, PCRCL = self.forward_encoder_decoder(x1_ori, x2_ori, x1_aug, x2_aug, gt)
            return out_ori, out_aug, aux_out, ACML1, ACML2, CCDL1, CCDL2, STCCL, PCRCL
        else:
            out_ori = self.forward_encoder_decoder(x1_ori, x2_ori)
            return out_ori

    def forward_encoder_decoder(self, x1_ori, x2_ori, x1_aug=None, x2_aug=None, gt=None):
        ori_arr1 = []
        aug_arr1 = []
        ori_arr2 = []
        aug_arr2 = []

        orid_arr = []
        augd_arr = []
        x_size = x1_ori.size()

        x1_ori = self.layer0[0](x1_ori)
        x2_ori = self.layer0[0](x2_ori)
        if self.wt_layer[2] == 1 or self.wt_layer[2] == 2:
            x1_ori, ori1 = self.layer0[1](x1_ori)
            ori_arr1.append(ori1)
            x2_ori, ori2 = self.layer0[1](x2_ori)
            ori_arr2.append(ori2)
        else:
            x1_ori = self.layer0[1](x1_ori)
            x2_ori = self.layer0[1](x2_ori)
        x1_ori = self.layer0[2](x1_ori)
        x1_ori = self.layer0[3](x1_ori)
        x2_ori = self.layer0[2](x2_ori)
        x2_ori = self.layer0[3](x2_ori)

        if self.training:
            x1_aug = self.layer0[0](x1_aug)
            x1_aug, aug1 = self.layer0[1](x1_aug)
            aug_arr1.append(aug1)
            x1_aug = self.layer0[2](x1_aug)
            x1_aug = self.layer0[3](x1_aug)

            x2_aug = self.layer0[0](x2_aug)
            x2_aug, aug2 = self.layer0[1](x2_aug)
            aug_arr2.append(aug2)
            x2_aug = self.layer0[2](x2_aug)
            x2_aug = self.layer0[3](x2_aug)

        x_tuple1 = self.layer1([x1_ori, ori_arr1])
        low_level1 = x_tuple1[0]
        x_tuple1 = self.layer2(x_tuple1)
        x_tuple1 = self.layer3(x_tuple1)

        aux_out1 = x_tuple1[0]
        x_tuple1 = self.layer4(x_tuple1)
        x1_ori = x_tuple1[0]
        ori_arr1 = x_tuple1[1]
        orid_arr1 = x_tuple1[0]

        if self.training:
            x1_aug_tuple = self.layer1([x1_aug, aug_arr1])
            low_level1_aug = x1_aug_tuple[0]

            x1_aug_tuple = self.layer2(x1_aug_tuple)
            x1_aug_tuple = self.layer3(x1_aug_tuple)

            x1_aug_tuple = self.layer4(x1_aug_tuple)
            x1_aug = x1_aug_tuple[0]
            aug_arr1 = x1_aug_tuple[1]
            augd_arr1 = x1_aug_tuple[0]

        x_tuple2 = self.layer1([x2_ori, ori_arr2])
        low_level2 = x_tuple2[0]
        x_tuple2 = self.layer2(x_tuple2)
        x_tuple2 = self.layer3(x_tuple2)
        aux_out2 = x_tuple2[0]
        x_tuple2 = self.layer4(x_tuple2)
        x2_ori = x_tuple2[0]
        ori_arr2 = x_tuple2[1]
        orid_arr2 = x_tuple2[0]

        if self.training:
            orid_arr12 = self.stage1_Conv3(torch.cat([orid_arr1, orid_arr2], 1))
            orid_arr.append(orid_arr12)

        x_orid = self.stage1_Conv1(torch.cat([x1_ori, x2_ori], 1))
        low_level = self.stage1_Conv2(torch.cat([low_level1, low_level2], 1))
        x_orid = self.aspp(x_orid)
        dec0_up_ori = self.bot_aspp(x_orid)
        dec0_fine_ori = self.bot_fine(low_level)
        dec0_up_ori = Upsample(dec0_up_ori, low_level.size()[2:])
        dec0_ori = [dec0_fine_ori, dec0_up_ori]
        dec0_ori = torch.cat(dec0_ori, 1)

        if self.training:
            orid_arr.append(dec0_ori)
        dec1_ori = self.final1(dec0_ori)

        if self.training:
            orid_arr.append(dec1_ori)
        dec2_ori = self.final2(dec1_ori)

        main_out_ori = Upsample(dec2_ori, x_size[2:])
        out_ori = self.cls_head1(main_out_ori)

        if self.training:
            x2_aug_tuple = self.layer1([x2_aug, aug_arr2])
            low_level2_aug = x2_aug_tuple[0]
            x2_aug_tuple = self.layer2(x2_aug_tuple)
            x2_aug_tuple = self.layer3(x2_aug_tuple)
            x2_aug_tuple = self.layer4(x2_aug_tuple)
            x2_aug = x2_aug_tuple[0]
            aug_arr2 = x2_aug_tuple[1]
            augd_arr2 = x2_aug_tuple[0]
            augd_arr12 = self.stage1_Conv3(torch.cat([augd_arr1, augd_arr2], 1))
            augd_arr.append(augd_arr12)

            x_augd = self.stage1_Conv1(torch.cat([x1_aug, x2_aug], 1))
            low_level_aug = self.stage1_Conv2(torch.cat([low_level1_aug, low_level2_aug], 1))
            x_augd = self.aspp(x_augd)
            dec0_up_aug = self.bot_aspp(x_augd)
            dec0_fine_aug = self.bot_fine(low_level_aug)
            dec0_up_aug = Upsample(dec0_up_aug, low_level_aug.size()[2:])
            dec0_aug = [dec0_fine_aug, dec0_up_aug]
            dec0_aug = torch.cat(dec0_aug, 1)
            augd_arr.append(dec0_aug)
            dec1_aug = self.final1(dec0_aug)
            augd_arr.append(dec1_aug)
            dec2_aug = self.final2(dec1_aug)
            main_out_aug = Upsample(dec2_aug, x_size[2:])
            out_aug = self.cls_head1(main_out_aug)

            aux_out1 = self.dsn(aux_out1)
            aux_out2 = self.dsn(aux_out2)
            x = self.neck_layer.fusion(aux_out1, aux_out2, self.neck_layer.policy)
            aux_out = self.cls_head(x)


            ACML1, CCDL1 = self._mean_feature_constraint_losses(ori_arr1, aug_arr1)
            ACML2, CCDL2 = self._mean_feature_constraint_losses(ori_arr2, aug_arr2)


            _, predict_ori = torch.max(dec2_ori, 1)
            _, predict_aug = torch.max(dec2_aug, 1)
            STCCL, PCRCL = self._mean_decoder_contrastive_losses(augd_arr, orid_arr, gt, predict_ori, predict_aug)

            return out_ori, out_aug, aux_out, ACML1, ACML2, CCDL1, CCDL2, STCCL, PCRCL

        else:
            return out_ori

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

    def forward(self, x1_ori, x2_ori, x1_aug=None, x2_aug=None, gt=None):
        if self.training == True:
            out_ori, out_aug, aux_out, ACML1, ACML2, CCDL1, CCDL2, STCCL, PCRCL = self.encoder_decoder(x1_ori, x2_ori, x1_aug, x2_aug, gt)
            return out_ori, out_aug, aux_out, ACML1, ACML2, CCDL1, CCDL2, STCCL, PCRCL
        else:
            out_ori = self.encoder_decoder(x1_ori, x2_ori)
            return out_ori
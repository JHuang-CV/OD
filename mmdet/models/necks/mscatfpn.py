import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from ..registry import NECKS
from ..utils import ConvModule
from mmdet.ops.context_block import ContextBlock

from mmdet.models.plugins.squeeze_excitation import ChannelSELayer


@NECKS.register_module
class MSCATFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None):
        super(MSCATFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = activation
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False

        self.epsilon = 1e-4

        self.se = ChannelSELayer(768)

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.cat_convs = nn.ModuleList()
        self.add_convs = nn.ModuleList()
        #self.gc_block = nn.ModuleList()

        self.relu = nn.ReLU()

        self.gc_block1 = ContextBlock(inplanes=256, ratio=1./4.)
        self.gc_block2 = ContextBlock(inplanes=256, ratio=1. / 4.)

        self.scat_conv = ConvModule(
                out_channels * (self.backbone_end_level-self.start_level),
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)

        self.c3_w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.c4_w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.c5_w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                activation=self.activation,
                inplace=False)
            cat_conv = ConvModule(
                out_channels * (self.backbone_end_level-self.start_level),
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            add_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                activation=self.activation,
                inplace=False)
            self.cat_convs.append(cat_conv)
            self.lateral_convs.append(l_conv)
            self.add_convs.append(add_conv)

            #self.gc_block.append(ContextBlock(inplanes=256, ratio=1./4.))
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        used_backbone_levels = len(laterals)

        mulscale_per_level = []
        for i in range(used_backbone_levels):
            level = []
            m = i - 0
            n = used_backbone_levels - 1 - i
            level.append(laterals[i])
            for x in range(m):
                level.insert(0, F.interpolate(level[0], scale_factor=2, mode='nearest'))
            for y in range(n):
                level.append(F.max_pool2d(level[-1], 2, stride=2))
            mulscale_per_level.append(level)
        sglscale_per_level = list(zip(*mulscale_per_level))
        feat_cat = [torch.cat(scale, 1)for scale in sglscale_per_level]
        #channel_se = [self.se(cat_ft) for cat_ft in feat_cat]
        mcat = [cat_conv(feat_cat[i]) for i, cat_conv in enumerate(self.cat_convs)]
        #outs = [gc(outs[i]) for i, gc in enumerate(self.gc_block)]
        mcat = [self.gc_block1(ft) for ft in mcat]

        single_list = []
        level = used_backbone_levels // 2

        for i in range(used_backbone_levels):
            if i < level:
                single_list.append(F.max_pool2d(laterals[i], 2, stride=2))
            elif i == level:
                single_list.append(laterals[i])
            else:
                single_list.append(F.interpolate(laterals[i], scale_factor=2, mode='nearest'))

        single_cat = torch.cat(single_list, 1)
        single_cat = self.scat_conv(single_cat)
        single_cat = self.gc_block2(single_cat)

        m = level - 0
        n = used_backbone_levels - 1 - level
        scat = [single_cat]
        for x in range(m):
            scat.insert(0, F.interpolate(scat[0], scale_factor=2, mode='nearest'))
        for y in range(n):
            scat.append(F.max_pool2d(scat[-1], 2, stride=2))

        # outs = [scat[i]+lateral for i, lateral in enumerate(laterals)]
        # outs = [add_conv(outs[i]) for i, add_conv in enumerate(self.add_convs)]

        outs = []
        for i, (m, s, l) in enumerate(zip(mcat, scat, laterals)):
             outs.append(
                 self.add_convs[i](m.sigmoid()*s/2 + l / 2)
             )

        if self.num_outs > used_backbone_levels:
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[0](orig))
                else:
                    outs.append(self.fpn_convs[0](outs[-1]))
                for i in range(1, self.num_outs-used_backbone_levels):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)

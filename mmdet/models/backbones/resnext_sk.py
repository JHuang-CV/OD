import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer
from .resnet import Bottleneck as _Bottleneck
from .resnet import ResNet


class SKConv(nn.Conv2d):
    def __init__(self, channels, stride, M=2, reduction=16):
        super().__init__(
            channels,
            channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=1,
            bias=False
        )
        # self.conv1 = nn.ModuleList([])
        # for i in range(M):
        #     self.conv1.append(nn.Sequential(
        #         nn.Conv2d(channels, channels, kernel_size=3, padding=1+i, dilation=1+i, bias=False),
        #         nn.BatchNorm2d(channels),
        #         nn.ReLU()
        #     ))
        self.weight_diff = nn.Parameter(torch.Tensor(self.weight.size()))
        self.weight_diff.data.zero_()
        self.M = M

        self.att_c = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels * M, 1, bias=False)
        )
        self.att_s = nn.Sequential(
            nn.Conv2d(1, 2, 3, 1, 1, bias=False)
        )
        self.att_3d = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        splited = list()
        splited.append(super().conv2d_forward(x, self.weight))
        for i in range(1, self.M):
            self.padding = 1+i
            self.dilation = 1+i
            weight = self.weight + self.weight_diff
            splited.append(super().conv2d_forward(x, weight))

        self.padding = 1
        self.dilation = 1
        
        feats = sum(splited)
        att_c = self.att_c(feats)
        att_c = att_c.view(x.size(0), self.M, x.size(1))
        att_c = F.softmax(att_c, dim=1)
        att_c = att_c.view(x.size(0), -1, 1, 1)
        att_c = torch.split(att_c, x.size(1), dim=1)

        feats_c = sum([w * s for w, s in zip(att_c, splited)])

        att_s = self.att_s(torch.mean(feats, dim=1, keepdim=True))
        att_s = F.softmax(att_s, dim=1)
        att_s = torch.split(att_s, 1, dim=1)

        feats_s = sum([w * s for w, s in zip(att_s, splited)])

        return (feats_c + feats_s)/2
        # return torch.where(feats_c > feats_s, feats_c, feats_s)
        #
        # att_3d = self.att_3d(feats_c+feats_s)
        #
        # return att_3d * feats_c + (1-att_3d) * feats_s


class Bottleneck(_Bottleneck):

    def __init__(self, inplanes, planes, groups=1, base_width=4, **kwargs):
        """Bottleneck block for ResNeXt.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(inplanes, planes, **kwargs)

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes * (base_width / 64)) * groups

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg, width, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg, width, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg, self.planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            # self.conv2 = build_conv_layer(
            #     self.conv_cfg,
            #     width,
            #     width,
            #     kernel_size=3,
            #     stride=self.conv2_stride,
            #     padding=self.dilation,
            #     dilation=self.dilation,
            #     groups=groups,
            #     bias=False)
            self.conv2 = SKConv(channels=width, stride=self.conv2_stride)

        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                self.dcn,
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width,
            self.planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   groups=1,
                   base_width=4,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   dcn=None,
                   gcb=None):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1],
        )

    layers = []
    layers.append(
        block(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            groups=groups,
            base_width=base_width,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn,
            gcb=gcb))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=1,
                dilation=dilation,
                groups=groups,
                base_width=base_width,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNeXtSK(ResNet):
    """ResNeXt backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.
        num_stages (int): Resnet stages, normally 4.
        groups (int): Group of resnext.
        base_width (int): Base width of resnext.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import ResNeXt
        >>> import torch
        >>> self = ResNeXt(depth=50)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 256, 8, 8)
        (1, 512, 4, 4)
        (1, 1024, 2, 2)
        (1, 2048, 1, 1)
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, groups=1, base_width=4, **kwargs):
        super(ResNeXtSK, self).__init__(**kwargs)
        self.groups = groups
        self.base_width = base_width

        self.inplanes = 64
        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            gcb = self.gcb if self.stage_with_gcb[i] else None
            planes = 64 * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                groups=self.groups,
                base_width=self.base_width,
                style=self.style,
                with_cp=self.with_cp,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                dcn=dcn,
                gcb=gcb)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

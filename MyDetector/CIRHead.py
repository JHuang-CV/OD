import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.models.registry import HEADS
from mmdet.models.utils import ConvModule, bias_init_with_prob
from mmdet.models.anchor_heads.anchor_head import AnchorHead
from mmdet.models.builder import build_loss
from mmdet.core import (AnchorGenerator, delta2bbox, force_fp32,
                        multi_apply, multiclass_nms)
from mmdet.core.bbox import PseudoSampler, assign_and_sample, bbox2delta, build_assigner

from .max_iou_assigner import MaxIoUAssigner

@HEADS.register_module
class CIRHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 anchor_ratios=[0.5, 1.0, 2.0],
                 anchor_strides=[4, 8, 16, 32, 64],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=0.11, loss_weight=1.0),
                 loss_IoUness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_softcls=dict(
                     type='MSELoss',
                     loss_weight=1.0)):

        super(CIRHead, self).__init__()
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

        if self.cls_out_channels <= 0:
            raise ValueError('num_classes={} is too small'.format(num_classes))

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_IoUness)
        self.loss_softcls = build_loss(loss_softcls)
        self.fp16_enabled = False

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, anchor_scales, anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.CIR_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.CIR_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 4, 3, padding=1)
        self.CIR_IoUness = nn.Conv2d(self.feat_channels, self.num_anchors, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.CIR_cls, std=0.01, bias=bias_cls)
        normal_init(self.CIR_reg, std=0.01)
        normal_init(self.CIR_IoUness, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.CIR_cls(cls_feat)
        bbox_pred = self.CIR_reg(reg_feat)
        IoU_feat = cls_feat+reg_feat
        IoUness_pred = self.CIR_IoUness(IoU_feat)
        return cls_score, bbox_pred, IoUness_pred

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i], device=device)
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w),
                    device=device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def loss_single(self, cls_score, bbox_pred, iou_pred, labels, label_weights,
                    bbox_targets, bbox_weights, iou_targets, iou_weights, softcls,
                    softcls_weights,num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        iou_targets = iou_targets.reshape(-1)
        iou_weights = iou_weights.reshape(-1)
        iou_pred = iou_pred.permute(0, 2, 3, 1).reshape(-1)
        loss_iou = self.loss_iou(iou_pred, iou_targets, iou_weights, avg_factor=num_total_samples)

        softcls = softcls.reshape(-1, self.cls_out_channels)
        softcls_weights = softcls_weights.reshape(-1, self.cls_out_channels)
        loss_softcls = self.loss_softcls(cls_score, softcls, softcls_weights, avg_factor = num_total_samples)

        return loss_cls, loss_bbox, loss_iou, loss_softcls

    def loss(self,
             cls_scores,
             bbox_preds,
             iou_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_iou_targets = self.anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_iou_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         IoU_targets_list, IoU_weights_list, num_total_pos, num_total_neg, softcls_list,
         softcls_weights_list) = cls_reg_iou_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)
        losses_cls, losses_bbox, losses_iou, losses_softcls = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            iou_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            IoU_targets_list,
            IoU_weights_list,
            softcls_list,
            softcls_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        # return dict(loss_cls=losses_cls, loss_bbox=losses_bbox,
        #             loss_iou=losses_iou, loss_softcls=losses_softcls)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

    def anchor_target(self,
                      anchor_list,
                      valid_flag_list,
                      gt_bboxes_list,
                      img_metas,
                      target_means,
                      target_stds,
                      cfg,
                      gt_bboxes_ignore_list=None,
                      gt_labels_list=None,
                      label_channels=1,
                      sampling=True,
                      unmap_outputs=True):
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights, all_IoU_targets, all_IoU_weights,
         pos_inds_list, neg_inds_list, all_softcls, all_softcls_weights) = multi_apply(
            self.anchor_target_single,
            anchor_list,
            valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            target_means=target_means,
            target_stds=target_stds,
            cfg=cfg,
            label_channels=label_channels,
            sampling=sampling,
            unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        # new add
        IoU_targets_list = images_to_levels(all_IoU_targets, num_level_anchors)
        IoU_weights_list = images_to_levels(all_IoU_weights, num_level_anchors)

        softcls_list = images_to_levels(all_softcls, num_level_anchors)
        softcls_weights_list = images_to_levels(all_softcls_weights, num_level_anchors)
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, IoU_targets_list, IoU_weights_list,
                num_total_pos, num_total_neg, softcls_list, softcls_weights_list)

    def anchor_target_single(self,
                             flat_anchors,
                             valid_flags,
                             gt_bboxes,
                             gt_bboxes_ignore,
                             gt_labels,
                             img_meta,
                             target_means,
                             target_stds,
                             cfg,
                             label_channels=1,
                             sampling=True,
                             unmap_outputs=True):
        inside_flags = self.anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           cfg.allowed_border)
        if not inside_flags.any():
            return (None,) * 6
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        if sampling:
            assign_result, sampling_result = assign_and_sample(
                anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
        else:
            bbox_assigner = MaxIoUAssigner()
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                                 gt_bboxes_ignore, gt_labels)
            bbox_sampler = PseudoSampler()
            sampling_result = bbox_sampler.sample(assign_result, anchors,
                                                  gt_bboxes)
        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        # new add
        IoU_targets = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        IoU_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        softcls_targets = anchors.new_zeros((num_valid_anchors, self.cls_out_channels), dtype=torch.float)
        softcls_weights = anchors.new_zeros((num_valid_anchors, self.cls_out_channels), dtype=torch.float)
        soft_gt_labels = sampling_result.soft_gt_labels
        unassigned_bboxes_inds = sampling_result.unassigned_bboxes_inds
        softcls_targets[unassigned_bboxes_inds, soft_gt_labels[unassigned_bboxes_inds, 0]-1] = \
            soft_gt_labels[unassigned_bboxes_inds, 1]
        softcls_weights[unassigned_bboxes_inds, :] = 1

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox2delta(sampling_result.pos_bboxes,
                                          sampling_result.pos_gt_bboxes,
                                          target_means, target_stds)
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = cfg.pos_weight
            # new add
            IoU_targets[pos_inds] = sampling_result.pos_gt_IoU
            IoU_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(labels, num_total_anchors, inside_flags)
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
            # new add
            IoU_targets = unmap(IoU_targets, num_total_anchors, inside_flags)
            IoU_weights = unmap(IoU_weights, num_total_anchors, inside_flags)

            softcls_targets = unmap(softcls_targets, num_total_anchors, inside_flags)
            softcls_weights = unmap(softcls_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, IoU_targets, IoU_weights, pos_inds,
                neg_inds, softcls_targets, softcls_weights)

    def anchor_inside_flags(self, flat_anchors, valid_flags, img_shape,
                            allowed_border=0):
        img_h, img_w = img_shape[:2]
        if allowed_border >= 0:
            inside_flags = valid_flags & \
                           (flat_anchors[:, 0] >= -allowed_border).type(torch.uint8) & \
                           (flat_anchors[:, 1] >= -allowed_border).type(torch.uint8) & \
                           (flat_anchors[:, 2] < img_w + allowed_border).type(torch.uint8) & \
                           (flat_anchors[:, 3] < img_h + allowed_border).type(torch.uint8)
        else:
            inside_flags = valid_flags
        return inside_flags


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret


def images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets
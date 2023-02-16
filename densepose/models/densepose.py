import torchvision as tv
import torchvision.models.detection.backbone_utils

import pickle
import collections
import numpy as np

import csv
import PIL.Image
import PIL.ImageDraw
import torch.nn.functional
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops import MultiScaleRoIAlign

from .msra_resnet import msra_resnet101
from .extra_fpn_block import PanopticExtraFPNBlock
from .roi_heads import DensePoseRoIHeads, DensePoseHead, DensePosePredictor
from .rpn import CustomRegionProposalNetwork
from .transform import DensePoseRCNNTransform
import torch.nn as nn
from torch import Tensor


# class DensePose(nn.Module):
#     def __init__(self, pretrained=True, backbone=None,
def densepose(pretrained=True, backbone=None,
        min_size=(640, 672, 704, 736, 768, 800), max_size=1333,
        image_mean=None, image_std=None,
        # RPN parameters
        rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
        box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512, box_positive_fraction=0.25,
        bbox_reg_weights=None):
    num_classes = 2  # Background: 0, Person: 1
    panoptic_out_channels = 256

    if pretrained:
        # HACK: Temporary add MSRA ResNet to torchvison's ResNets list
        backbone_name = 'msra_resnet101'
        tv.models.resnet.__dict__[backbone_name] = msra_resnet101
        backbone = tv.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name=backbone_name,
            pretrained=False,
            trainable_layers=5,
            extra_blocks=PanopticExtraFPNBlock(
                featmap_names=['0', '1', '2', '3'],
                in_channels=256,
                out_channels=panoptic_out_channels,
                conv_dims=256
                )
            )
        tv.models.resnet.__dict__.pop(backbone_name)

        image_mean = [123.675, 116.280, 103.530][::-1]
        image_std = [1.0, 1.0, 1.0]

    if backbone is None:
        backbone = tv.models.detection.backbone_utils.resnet_fpn_backbone(
            backbone_name='resnet101',
            pretrained=False,
            trainable_layers=5,
            extra_blocks=PanopticExtraFPNBlock(
                featmap_names=['0', '1', '2', '3'],
                in_channels=256,
                out_channels=panoptic_out_channels,
                conv_dims=256
                )
            )

    if not hasattr(backbone, "out_channels"):
        raise ValueError(
            "backbone should contain an attribute out_channels "
            "specifying the number of output channels (assumed to be the "
            "same for all the levels)")

    backbone_out_channels = backbone.out_channels

    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(
        anchor_sizes, aspect_ratios
        )

    rpn_head = RPNHead(
        backbone_out_channels, rpn_anchor_generator.num_anchors_per_location()[0]
        )

    rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
    rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train,
                              testing=rpn_post_nms_top_n_test)

    rpn = CustomRegionProposalNetwork(
        ['0', '1', '2', '3', 'pool'],
        rpn_anchor_generator, rpn_head,
        rpn_fg_iou_thresh, rpn_bg_iou_thresh,
        rpn_batch_size_per_image, rpn_positive_fraction,
        rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
        score_thresh=rpn_score_thresh
        )

    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
        )
    box_resolution = box_roi_pool.output_size[0]
    box_representation_size = 1024
    box_head = TwoMLPHead(
        backbone_out_channels * box_resolution ** 2,
        box_representation_size)
    box_predictor = FastRCNNPredictor(
        box_representation_size,
        num_classes)

    densepose_roi_pool = MultiScaleRoIAlign(
        featmap_names=['panoptic_feature'],
        output_size=28,
        sampling_ratio=2
        )
    densepose_head_out_channels = 512
    densepose_head = DensePoseHead(
        panoptic_out_channels,
        densepose_head_out_channels,
        )
    densepose_predictor = DensePosePredictor(
        in_channels=densepose_head_out_channels,
        num_segmentations=2,
        num_patches=24,
        kernel_size=(4, 4),
        scale_factor=2
        )

    roi_heads = DensePoseRoIHeads(
        box_roi_pool, box_head, box_predictor,
        box_fg_iou_thresh, box_bg_iou_thresh,
        box_batch_size_per_image, box_positive_fraction,
        bbox_reg_weights,
        box_score_thresh, box_nms_thresh, box_detections_per_img,
        densepose_roi_pool=densepose_roi_pool,
        densepose_head=densepose_head,
        densepose_predictor=densepose_predictor
        )

    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225]
    transform = DensePoseRCNNTransform(min_size, max_size, image_mean, image_std)

    model = GeneralizedRCNN(backbone, rpn, roi_heads, transform)
    return model

    # def forward(self, x: Tensor):
    #     return model(x)

    # for key in list(state_dict.keys()):
    #     state_dict[key.replace("rpn.rpn.head.conv.0.0.weight", "rpn.head.conv.weight")
    #         .replace("rpn.rpn.head.conv.0.0.bias","rpn.head.conv.bias")
    #         .replace("rpn.rpn.head.cls_logits.weight","rpn.head.cls_logits.weight")
    #         .replace("rpn.rpn.head.cls_logits.bias","rpn.head.cls_logits.bias")
    #         .replace("rpn.rpn.head.bbox_pred.weight","rpn.head.bbox_pred.weight")
    #         .replace("rpn.rpn.head.bbox_pred.bias","rpn.head.bbox_pred.bias")] = state_dict.pop(key)

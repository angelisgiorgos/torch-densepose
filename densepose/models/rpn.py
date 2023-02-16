from collections import OrderedDict
from typing import Dict, Optional, List, Tuple
import torch
from torch import Tensor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
import torch.nn as nn


class CustomRegionProposalNetwork(nn.Module):
    def __init__(
            self, featmap_names: List[str], anchor_generator: AnchorGenerator,
            head: RPNHead, fg_iou_thresh: float, bg_iou_thresh: float,
            batch_size_per_image: int, positive_fraction: float,
            pre_nms_top_n: Dict[str, int], post_nms_top_n: Dict[str, int], nms_thresh: float,
            score_thresh: float = 0.0
            ):
        super().__init__()
        self.rpn = RegionProposalNetwork(anchor_generator, head, fg_iou_thresh,
            bg_iou_thresh, batch_size_per_image, positive_fraction,
            pre_nms_top_n, post_nms_top_n, nms_thresh, score_thresh)

        self.featmap_names = featmap_names

    def forward(self,
                images: ImageList,
                features: Dict[str, Tensor],
                targets: Optional[List[Dict[str, Tensor]]] = None
                ) -> Tuple[List[Tensor], Dict[str, Tensor]]:

        filtered_features = OrderedDict()
        for k, v in features.items():
            if k in self.featmap_names:
                filtered_features[k] = v

        return self.rpn(images, filtered_features, targets)


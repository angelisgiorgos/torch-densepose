from torch import Tensor
import torch.nn as nn
import torch
from typing import Tuple, List, Optional, Callable

from torchvision.ops.feature_pyramid_network import LastLevelMaxPool, ExtraFPNBlock


class PanopticExtraFPNBlock(ExtraFPNBlock):
    def __init__(self, featmap_names: List[str], in_channels: int, out_channels: int,
                 conv_dims: int,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(PanopticExtraFPNBlock, self).__init__()

        self.featmap_names = featmap_names

        featmap_ids = [int(i) for i in featmap_names]

        self.predictor = nn.Conv2d(conv_dims, out_channels, kernel_size=(1, 1), stride=(1, 1),
                                   padding=(0, 0))
        self.blocks = {}

        highest_resolution_featmap_id = featmap_ids[0]
        for featmap_name, featmap_id in zip(featmap_names, featmap_ids):
            ops = []
            num_upsampling = featmap_id - highest_resolution_featmap_id
            num_convs = max(1, num_upsampling)
            require_upsampling = num_upsampling > 0

            for j in range(num_convs):
                conv = nn.Conv2d(
                    in_channels if j == 0 else conv_dims,
                    conv_dims,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    bias=norm_layer is None,
                    )
                ops.append(conv)

                if norm_layer is not None:
                    ops.append(norm_layer(conv_dims))

                relu = nn.ReLU()
                ops.append(relu)

                if require_upsampling:
                    upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                    ops.append(upsampling)

            block = nn.Sequential(*ops)
            self.add_module('block{}'.format(featmap_id), block)

            self.last_level_pool = LastLevelMaxPool()

    # @torch.jit.script_method
    # def list_creation(self, results: List[Tensor], features: List[str], names: List[str]) -> \
    #         Tuple[List[int],
    #               List[torch.Tensor]]:
    #     keys_list = []
    #     for k in names:
    #         if k in features:
    #             keys_list.append(int(k))
    #
    #     values_list = []
    #     for k, v in zip(names, results):
    #         if k in features:
    #             values_list.append(v)
    #     return keys_list, values_list

    def _get_layer_name(self, i: int):
        layer_name = "block{}".format(i)
        return layer_name

    def forward(
            self,
            results: List[Tensor],
            x: List[Tensor],
            names: List[str],
            ) -> Tuple[List[Tensor], List[str]]:
        # TODO: TorchScript support
        blocks: torch.Tensor = torch.zeros(1)

        results, names = self.last_level_pool(results, x, names)

        # print(type(names), type(self.featmap_names))

        keys: List[int] = []
        for k in names:
            # print(type(k))
            if k in self.featmap_names:
                keys.append(int(k))

        values: List[Tensor] = []
        for k, v in zip(names, results):
            if k in self.featmap_names:
                values.append(v)

        out = getattr(self, 'block0')(values[0])

        for k, v in zip(keys[1:], values[1:]):
            if str(k) == '1':
                blocks = getattr(self, 'block1')(v)
            elif str(k) == '2':
                blocks = getattr(self, 'block2')(v)
            elif str(k) == '3':
                blocks = getattr(self, 'block3')(v)
            out = out + blocks

        results.append(self.predictor(out))
        names.append('panoptic_feature')

        return results, names

import torch
import functools
import torch.nn as nn

from ...utils.spconv_utils import spconv
from ..model_utils.unet_blocks import ResidualBlock, VGGBlock, UBlock
from ...utils import common_utils


class SparseUNetIndoor(nn.Module):
    def __init__(self, model_cfg):
        super(SparseUNetIndoor, self).__init__()
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        self.model_cfg = model_cfg
        self.in_channel = model_cfg.IN_CHANNEL
        self.mid_channel = model_cfg.MID_CHANNEL
        self.block_reps = model_cfg.BLOCK_REPS
        self.block_residual = model_cfg.BLOCK_RESIDUAL
        self.num_blocks = model_cfg.get('NUM_BLOCKS', None)
        self.num_filters = model_cfg.get('NUM_FILTERS', None)

        if self.block_residual:
            block = functools.partial(ResidualBlock, custom_sp1x1=self.model_cfg.get('CUSTOM_SP1X1', False))
        else:
            block = VGGBlock

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                self.in_channel, self.mid_channel, kernel_size=3, padding=1, bias=False, indice_key='subm1'
            )
        )

        if self.num_filters is not None:
            block_channels = self.num_filters
        else:
            assert self.num_blocks is not None
            block_channels = [self.mid_channel * (i + 1) for i in range(self.num_blocks)]

        self.unet = UBlock(block_channels, norm_fn, self.block_reps, block, indice_key_id=1)
        self.output_layer = spconv.SparseSequential(
            norm_fn(self.mid_channel), nn.ReLU()
        )

        # init parameters
        self.apply(self.set_bn_init)

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def forward(self, batch_dict):
        input_sp_tensor = spconv.SparseConvTensor(
            batch_dict['voxel_features'], batch_dict['voxel_coords'].int(),
            batch_dict['spatial_shape'], batch_dict['batch_size']
        )
        output = self.input_conv(input_sp_tensor)
        output = self.unet(output)
        output = self.output_layer(output)
        output_feats = output.features
        # if not self.training and batch_dict['test_x4_split']:
        #     output_feats = common_utils.merge_4_parts(output_feats)

        batch_dict['backbone_3d_feats'] = output_feats
        return batch_dict

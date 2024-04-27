import functools
import torch.nn as nn
import numpy as np

from pcseg.utils import loss_utils
from pcseg.models.model_utils.basic_block_1d import build_block
from ..model_utils import basic_block_1d


class KDHeadTemplate(nn.Module):
    def __init__(self, model_cfg) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.in_feature_name = model_cfg.IN_FEAT_NAME
        # self.out_feature_name = model_cfg.OUT_FEAT_NAME
        
        # self.kd_adapter = BasicAdaptLayer(model_cfg.KD_ADAPTER)

        self.feature_norm = model_cfg.FEAT_NORM
        
        self.loss_func = loss_utils.CosineSimilarityLoss()
        self.forward_ret_dict = {}
    
    def forward(self, batch_dict):
        self.forward_ret_dict = {}
        # out_feature = self.kd_adapter(batch_dict[self.in_feature_name])
        feature = batch_dict[self.in_feature_name]
        out_feature = feature[batch_dict['v2p_map'].long()]

        if self.feature_norm:
            out_feature = out_feature / out_feature.norm(dim=-1, keepdim=True)
        
        # batch_dict[self.out_feature_name] = out_feature
        if self.training:
            self.forward_ret_dict['output'] = out_feature
            self.forward_ret_dict['kd_labels'] = batch_dict['kd_labels']
            self.forward_ret_dict['kd_labels_mask'] = batch_dict['kd_labels_mask']
        return batch_dict
    
    def get_loss(self):
        pred = self.forward_ret_dict['output']
        target = self.forward_ret_dict['kd_labels']
        mask = self.forward_ret_dict['kd_labels_mask']
        if target.shape[0] == mask.shape[0]:   
            kd_loss = self.loss_func(pred, target[mask], mask)
        else:
            assert target.shape[0] == mask.sum().item()
            kd_loss = self.loss_func(pred, target, mask)

        tb_dict = {'loss_kd': kd_loss.item()}
        return kd_loss, tb_dict


class BasicAdaptLayer(nn.Module):
    def __init__(self, block_cfg):
        super().__init__()
        self.block_cfg = block_cfg

        self.build_adaptation_layer(block_cfg)
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def build_adaptation_layer_old(self, block_cfg):
        align_block = []

        in_channel = block_cfg.in_channel
        block_types = block_cfg.block_type
        num_filters = block_cfg.num_filters
        kernel_sizes = block_cfg.kernel_size
        num_strides = block_cfg.strides
        paddings = block_cfg.padding

        for i in range(len(num_filters)):
            align_block.extend(build_block(
                block_types[i], in_channel, num_filters[i], kernel_size=kernel_sizes[i],
                stride=num_strides[i], padding=paddings[i], bias=False
            ))
        self.adapt_layer = nn.Sequential(*align_block)

    def build_adaptation_layer(self, block_cfg):
        in_channel = block_cfg.in_channel
        out_channel = block_cfg.num_filters[0]
        num_adapter_layers = block_cfg.num_layers

        if num_adapter_layers == 1:
            mid_channel_list = [in_channel, out_channel]
        elif num_adapter_layers == 2:
            multiplier = int(np.log2(out_channel / in_channel))
            mid_channel_list = [in_channel, in_channel * multiplier, out_channel]
        else:
            raise NotImplementedError

        self.adapt_layer = basic_block_1d.MLP(
            mid_channel_list,
            norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
            num_layers=num_adapter_layers, last_norm_fn=False
        )

    def forward(self, in_feature):
        out_feature = self.adapt_layer(in_feature)
        return out_feature

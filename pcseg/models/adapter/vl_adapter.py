import functools

import torch.nn as nn
import numpy as np

from ..model_utils import basic_block_1d


class VLAdapter(nn.Module):
    def __init__(self, model_cfg, in_channel):
        super(VLAdapter, self).__init__()
        self.model_cfg = model_cfg
        self.in_feature_name = model_cfg.get('IN_FEAT_NAME', 'backbone_3d_feats')
        self.eval_only = model_cfg.get('EVAL_ONLY', None)
        self.text_channel = model_cfg.TEXT_DIM
        
        # vision adapter
        adapter_last_norm = self.model_cfg.get('LAST_NORM', True)
        self.adapter = self.build_vl_adapter(self.model_cfg.NUM_ADAPTER_LAYERS, in_channel, adapter_last_norm)
    
    def build_vl_adapter(self, num_adapter_layers, in_channel, last_norm):
        """build vision language adapter

        Args:
            num_adapter_layers (_type_): _description_
            in_channel (_type_): _description_

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        if num_adapter_layers < 1 or self.eval_only:
            return None

        if num_adapter_layers == 1:
            mid_channel_list = [in_channel, self.text_channel]
        elif num_adapter_layers == 2:
            multiplier = int(np.log2(self.text_channel / in_channel))
            mid_channel_list = [in_channel, in_channel * multiplier, self.text_channel]
        else:
            raise NotImplementedError

        adapter = basic_block_1d.MLP(
            mid_channel_list,
            norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
            num_layers=num_adapter_layers, last_norm_fn=last_norm
        )
        return adapter

    def forward(self, batch_dict):
        if self.eval_only and self.training:
            return batch_dict
        
        backbone3d_feats = batch_dict[self.in_feature_name]

        # forward adapter
        if hasattr(self, 'adapter') and self.adapter is not None:
            adapter_feats = self.adapter(backbone3d_feats)
        else:
            adapter_feats = backbone3d_feats

        batch_dict['adapter_feats'] = adapter_feats
        return batch_dict

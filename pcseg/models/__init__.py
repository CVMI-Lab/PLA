from collections import namedtuple

import numpy as np
import torch

from .vision_networks import build_model
from .text_networks import build_text_model

try:
    import kornia
except:
    pass 
    # print('Warning: kornia is not installed. This package is only required by CaDDN')


def build_vision_network(model_cfg, num_class, dataset):
    model = build_model(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def build_text_network(model_cfg):
    text_encoder = build_text_model(model_cfg=model_cfg)
    return text_encoder


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if isinstance(val, torch.Tensor):
            batch_dict[key] = batch_dict[key].cuda()
        elif not isinstance(val, np.ndarray) or key in ['calib', 'point_img_idx', 'point_img']:
            continue
        elif key in ['ids', 'metadata', 'scene_name']:
            continue
        elif key in ['points_xyz_voxel_scale', 'labels', 'inst_label', 'origin_idx', 'offsets', 'inst_cls']:
            batch_dict[key] = torch.from_numpy(val).long().cuda()
        elif key in ['inst_pointnum', 'batch_idxs']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()

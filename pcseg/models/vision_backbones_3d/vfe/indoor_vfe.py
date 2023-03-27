import torch

from .vfe_template import VFETemplate
from ....external_libs.softgroup_ops.ops import functions as sg_ops


class IndoorVFE(VFETemplate):
    def __init__(self, model_cfg, voxel_mode, **kwargs):
        super(IndoorVFE, self).__init__(model_cfg)
        self.use_xyz = model_cfg.get('USE_XYZ', False)
        self.voxel_mode = voxel_mode

    def forward(self, batch):
        batch_size = batch['batch_size']
        # voxelization
        # current implementation cannot support cuda
        # TODO: modify the voxelization part
        voxel_coords, v2p_map, p2v_map = sg_ops.voxelization_idx(
            batch['points_xyz_voxel_scale'].cpu(), batch_size, self.voxel_mode
        )
        voxel_coords, v2p_map, p2v_map = voxel_coords.cuda(), v2p_map.cuda(), p2v_map.cuda()

        feats = batch['feats']  # (N, C), float32, cuda

        voxel_feats = sg_ops.voxelization(feats, p2v_map, self.voxel_mode)

        batch.update({
            'voxel_features': voxel_feats,
            'v2p_map': v2p_map.long(),
            'voxel_coords': voxel_coords
        })

        return batch

import numpy as np
import torch

from .vfe_template import VFETemplate

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate


class DynamicMeanVFENoRange(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, **kwargs):
        super().__init__(model_cfg=model_cfg)
        self.num_point_features = num_point_features

        self.voxel_size = torch.tensor(voxel_size).cuda()

    def get_output_feature_dim(self):
        return self.num_point_features

    @torch.no_grad()
    def forward(self, batch_dict, **kwargs):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
            **kwargs:

        Returns:
            vfe_features: (num_voxels, C)
        """
        points = batch_dict['points']  # (batch_idx, x, y, z, i, e)

        assert points.min() >= 0

        # calculate the point cloud range.
        grid_size = torch.ceil((points[:, 1:4].max(0)[0] - points[:, 1:4].min(0)[0]) / self.voxel_size)
        scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        scale_yz = grid_size[1] * grid_size[2]
        scale_z = grid_size[2]

        # # debug
        point_coords = torch.floor((points[:, 1:4]) / self.voxel_size).int()

        merge_coords = points[:, 0].int() * scale_xyz + \
                       point_coords[:, 0] * scale_yz + \
                       point_coords[:, 1] * scale_z + \
                       point_coords[:, 2]
        points_data = points[:, 1:].contiguous()

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        points_mean = torch_scatter.scatter_mean(points_data, unq_inv, dim=0)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // scale_xyz,
                                    (unq_coords % scale_xyz) // scale_yz,
                                    (unq_coords % scale_yz) // scale_z,
                                    unq_coords % scale_z), dim=1)
        # voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['voxel_features'] = points_mean.contiguous()
        batch_dict['voxel_coords'] = voxel_coords.contiguous()
        batch_dict['v2p_map'] = unq_inv.contiguous()
        batch_dict['spatial_shape'] = grid_size.long() + 1

        return batch_dict

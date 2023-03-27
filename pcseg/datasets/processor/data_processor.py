import torch
import numpy as np

from functools import partial
from skimage import transform
from torch_scatter import scatter_mean

from ...utils import common_utils, voxelize_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, voxel_size, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        self.voxel_size = np.array(voxel_size, dtype=np.float32)
        if point_cloud_range is not None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / voxel_size
            self.grid_size = np.round(grid_size).astype(np.int64)
        else:
            self.grid_size = None

        for process_name in processor_configs.PROCESS_LIST:
            cur_processor = getattr(self, process_name)(config=processor_configs[process_name])
            self.data_processor_queue.append(cur_processor)

    def clip_points_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.clip_points_outside_range, config=config)

        if data_dict.get('points', None) is not None and self.point_cloud_range is not None:
            points_xyz = data_dict['points'][:, :3]
            points_xyz = np.clip(points_xyz, self.point_cloud_range[:3], self.point_cloud_range[3:])
            data_dict['points'][:, :3] = points_xyz

        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def custom_voxelization_indoor(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.custom_voxelization_indoor, config=config)

        xyz_voxel_scale = data_dict['points'][:, :3] / self.voxel_size
        xyz_voxel_scale -= xyz_voxel_scale.min(0)
        data_dict['points_xyz_voxel_scale'] = xyz_voxel_scale
        data_dict['feats'] = data_dict['points']
        data_dict['points_xyz'] = data_dict['points'][:, :3]

        return data_dict

    def custom_voxelization_one(self, data_dict=None, config=None):
        if data_dict is None:
            if self.training:
                return partial(self.custom_voxelization_one, config=config)
            else:
                return partial(self.custom_voxelization_mean, config=config)

        feats = data_dict['feats']
        coord = data_dict['points'][:, :3]
        labels = data_dict['labels']
        binary_labels = data_dict['binary_labels']
        xyz_norm = config['xyz_norm']

        # voxelization process
        coord_min = np.min(coord, 0)
        coord_norm = coord - coord_min

        uniq_idx, idx_recon = voxelize_utils.voxelize_with_rec_idx(coord_norm, self.voxel_size, training=True)
        coord_voxel = np.floor(coord_norm[uniq_idx] / np.array(self.voxel_size))
        coord, feats = coord[uniq_idx], feats[uniq_idx]

        if config.get('voxel_label', False):
            labels = labels[uniq_idx]
            binary_labels = binary_labels[uniq_idx]

        if xyz_norm:
            coord_min = np.min(coord, 0)
            coord -= coord_min

        coord_voxel = torch.LongTensor(coord_voxel)
        feats = torch.FloatTensor(feats)
        idx_recon = torch.LongTensor(idx_recon)

        data_dict['voxel_coords'] = coord_voxel
        data_dict['voxel_features'] = feats
        data_dict['labels'] = labels
        data_dict['v2p_map'] = idx_recon
        data_dict['binary_labels'] = binary_labels

        data_dict.pop('points_xyz_voxel_scale', None)
        data_dict.pop('feats', None)

        return data_dict

    def custom_voxelization_mean(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.custom_voxelization_mean, config=config)

        feats = data_dict['feats']
        coord = data_dict['points'][:, :3]
        labels = data_dict['labels']
        binary_labels = data_dict['binary_labels']
        xyz_norm = config['xyz_norm']

        # voxelization process
        coord_min = np.min(coord, 0)
        coord_norm = coord - coord_min

        uniq_idx, idx_recon = voxelize_utils.voxelize_with_rec_idx(coord_norm, self.voxel_size, training=False)

        if xyz_norm:
            coord_min = np.min(coord, 0)
            coord -= coord_min

        coord = torch.FloatTensor(coord)
        feats = torch.FloatTensor(feats)
        idx_recon = torch.LongTensor(idx_recon)

        coord_norm = coord - coord.min(0)[0]
        coord_norm = scatter_mean(coord_norm, idx_recon, dim=0)
        coord_norm = torch.floor(coord_norm / torch.from_numpy(self.voxel_size)).long()

        coord, feats = scatter_mean(coord, idx_recon, dim=0), scatter_mean(feats, idx_recon, dim=0)

        if self.training and config.get('voxel_label', False):
            labels = labels[uniq_idx]
            binary_labels = binary_labels[uniq_idx]

        data_dict['voxel_coords'] = coord_norm
        # data_dict['xyz'] = coord
        data_dict['voxel_features'] = feats
        data_dict['labels'] = labels
        data_dict['binary_labels'] = binary_labels
        data_dict['v2p_map'] = idx_recon

        data_dict.pop('points_xyz_voxel_scale', None)
        data_dict.pop('feats', None)

        return data_dict
        
    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        points = data_dict['points']
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict

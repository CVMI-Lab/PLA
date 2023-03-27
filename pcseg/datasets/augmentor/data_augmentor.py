from functools import partial

import numpy as np

from . import augmentor_utils


class DataAugmentor(object):
    def __init__(self, dataset_cfg, **kwargs):
        self.data_augmentor_queue = []
        self.aug_cfg = dataset_cfg.DATA_AUG
        self.kwargs = kwargs
        aug_config_list = self.aug_cfg.AUG_LIST

        self.data_augmentor_queue = []
        for aug in aug_config_list:
            if aug not in self.aug_cfg:
                continue
            cur_augmentor = partial(getattr(self, aug), config=self.aug_cfg[aug])
            self.data_augmentor_queue.append(cur_augmentor)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def shuffle(self, data_dict=None, config=None):
        shuffle_idx = np.random.permutation(data_dict['points_xyz'].shape[0])
        data_dict = self.update_data_dict(data_dict, shuffle_idx)
        return data_dict

    def crop(self, data_dict=None, config=None):
        data_dict['points_xyz_voxel_scale'], valid_idxs = augmentor_utils.crop(
            data_dict['points_xyz_voxel_scale'], self.kwargs['full_scale'], self.kwargs['max_npoint'], config.step,
        )
        data_dict = self.update_data_dict(data_dict, valid_idxs)
        if data_dict['points_xyz'].shape[0] == 0:
            data_dict['valid'] = False
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        data_dict['valid'] = True
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
        return data_dict

    def scene_aug(self, data_dict=None, config=None):
        if self.check_func(config) and self.check_data(data_dict):
            data_dict['points_xyz'], data_dict['rgb'] = augmentor_utils.scene_aug(
                config, data_dict['points_xyz'], data_dict['rgb']
            )
            if data_dict['points_xyz'].shape[0] == 0:
                data_dict['valid'] = False
        return data_dict

    @staticmethod
    def update_data_dict(data_dict, idx):
        for key in data_dict:
            if key in ['points_xyz', 'points', 'points_xyz_voxel_scale', 'rgb', 'labels',
                       'inst_label', 'binary_labels', 'origin_idx']:
                if data_dict[key] is not None:
                    data_dict[key] = data_dict[key][idx]
        return data_dict

    @staticmethod
    def check_func(key):
        return augmentor_utils.check_key(key) and augmentor_utils.check_p(key)

    def elastic(self, data_dict=None, config=None):
        data_dict['points_xyz_voxel_scale'] = data_dict['points_xyz'] * self.kwargs['voxel_scale']
        if self.check_func(config) and self.check_data(data_dict):
            for (gran_fac, mag_fac) in config.value:
                data_dict['points_xyz_voxel_scale'] = augmentor_utils.elastic(
                    data_dict['points_xyz_voxel_scale'], gran_fac * self.kwargs['voxel_scale'] // 50,
                    mag_fac * self.kwargs['voxel_scale'] / 50
                )
            if config.apply_to_feat:
                data_dict['points_xyz'] = data_dict['points_xyz_voxel_scale'] / self.kwargs['voxel_scale']

        # offset
        data_dict['points'] = data_dict['points_xyz_voxel_scale'] / self.kwargs['voxel_scale']
        data_dict['points_xyz_voxel_scale'] -= data_dict['points_xyz_voxel_scale'].min(0)
        return data_dict

    @staticmethod
    def check_data(data_dict):
        return ('valid' not in data_dict) or data_dict['valid']

    ###################
    # Used in outdoor #
    ###################
    @staticmethod
    def random_world_rotation(data_dict=None, config=None):
        points = data_dict['points']
        rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
        c, s = np.cos(rotate_rad), np.sin(rotate_rad)
        j = np.matrix([[c, s], [-s, c]])
        data_dict['points'][:, :2] = np.dot(points[:, :2], j)

        return data_dict

    @staticmethod
    def random_world_flip(data_dict=None, config=None):
        points = data_dict['points']
        flip_type = np.random.choice(4, 1)

        if flip_type == 0:
            # flip x only
            points[:, 0] = -points[:, 0]
        elif flip_type == 1:
            # flip y only
            points[:, 1] = -points[:, 1]
        elif flip_type == 2:
            # flip x+y
            points[:, :2] = -points[:, :2]

        data_dict['points'] = points
        return data_dict

    @staticmethod
    def random_world_scaling(data_dict=None, config=None):
        points = data_dict['points']
        noise_scale = np.random.uniform(config[0], config[1])
        points[:, :2] = noise_scale * points[:, :2]

        data_dict['points'] = points
        return data_dict

    @staticmethod
    def random_world_translation(data_dict=None, config=None):
        points = data_dict['points']
        noise_translate = np.array(
            [np.random.normal(0, config[0], 1), np.random.normal(0, config[1], 1), np.random.normal(0, config[2], 1)]
        ).T
        points[:, 0:3] += noise_translate

        data_dict['points'] = points
        return data_dict

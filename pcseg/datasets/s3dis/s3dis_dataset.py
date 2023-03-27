import SharedArray as SA
import os
import numpy as np
import pickle
import json
import torch

from ..indoor_dataset import IndoorDataset
from ...utils.common_utils import sa_create, sa_delete


class S3DISDataset(IndoorDataset):
    def __init__(self, dataset_cfg, class_names, training, root_path, logger=None):
        super(S3DISDataset, self).__init__(
            dataset_cfg, class_names, training, root_path, logger=logger
        )

        data_list = sorted(os.listdir(self.root_path / 'stanford_indoor3d_inst/'))
        data_list = [item[:-4] for item in data_list if 'Area_' in item]
        if dataset_cfg.DATA_SPLIT[self.mode] == 'train':
            self.data_list = [item for item in data_list if not 'Area_{}'.format(dataset_cfg.DATA_SPLIT.test_area) in item]
        else:
            self.data_list = [item for item in data_list if 'Area_{}'.format(dataset_cfg.DATA_SPLIT.test_area) in item]

        self.test_x4_split = dataset_cfg.DATA_PROCESSOR.x4_split

        self.put_data_to_shm()

        if self.training and hasattr(self, 'caption_cfg') and self.caption_cfg.get('CAPTION_CORR_PATH_IN_ONE_FILE', True):
            self.scene_image_corr_infos, self.scene_image_corr_entity_infos = self.include_caption_infos()

        self.load_image = self.dataset_cfg.get('LOAD_IMAGE', None)
        self.depth_image_scale = self.dataset_cfg.get('DEPTH_IMAGE_SCALE', None)
        self.image_path = self.dataset_cfg.get('IMAGE_PATH', None)

        if self.load_image:
            self.pc_mins = json.load(open(str(self.root_path / 'pc_min.json')))

        self.logger.info(
            "Totally {} samples in {} set.".format(
                len(self.data_list) * (self.repeat if self.training else 1), self.mode))

    def __len__(self):
        return len(self.data_list) * (self.repeat if self.training else 1)

    def put_data_to_shm(self):
        for item in self.data_list:
            if self.cache and not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(self.root_path / 'stanford_indoor3d_inst', item + '.npy')
                data = np.load(data_path)  # xyzrgbli, N*8
                sa_create("shm://{}".format(item), data)

    def load_data(self, index):
        fn = self.data_list[index]
        if self.cache:
            data = SA.attach("shm://{}".format(fn)).copy()
        else:
            data_path = os.path.join(self.root_path / 'stanford_indoor3d_inst', fn + '.npy')
            data = np.load(data_path)

        xyz_all, rgb_all, label_all, inst_label_all = data[:, 0:3], data[:, 3:6], data[:, 6], data[:, 7]

        # base / novel label
        if hasattr(self, 'base_class_mapper'):
            binary_label_all = self.binary_class_mapper[label_all.astype(np.int64)].astype(np.float32)
        else:
            binary_label_all = np.ones_like(label_all)
        if self.class_mode == 'base':
            label_all = self.base_class_mapper[label_all.astype(np.int64)]
        elif self.class_mode == 'novel':
            label_all = self.novel_class_mapper[label_all.astype(np.int64)]
        elif self.class_mode == 'all' and hasattr(self, 'ignore_class_idx'):
            label_all = self.valid_class_mapper[label_all.astype(np.int64)]
        inst_label_all[label_all == self.ignore_label] = self.ignore_label

        # # pseudo label
        # if self.training and self.pseudo_labels_dir is not None:
        #     pseudo_label_all = self.load_pseudo_labels(fn.split('/')[-1])
        #     if self.use_pseudo_gt_label:
        #         label_all[pseudo_label_all != label_all] = self.ignore_label
        #     else:
        #         label_all[binary_label_all == 0] = pseudo_label_all[binary_label_all == 0]
        return xyz_all, rgb_all, label_all, inst_label_all, binary_label_all

    def __getitem__(self, item):
        if (not self.training) and self.test_x4_split:
            if 'custom_voxelization_mean' in self.dataset_cfg.DATA_PROCESSOR.PROCESS_LIST:
                return self.get_test_item_vox_lai(item)
            else:
                return self.get_test_item(item)
        else:
            return self.get_train_item(item)

    def get_train_item(self, item):
        index = item % len(self.data_list)
        xyz_all, rgb_all, label_all, inst_label_all, binary_label_all = self.load_data(index)
        xyz_all -= xyz_all.mean(0)
        rgb_all = rgb_all / 127.5 - 1.0

        pc_count = xyz_all.shape[0]
        origin_idx = np.arange(xyz_all.shape[0]).astype(np.int64)
        # ==== caption ====
        scene_name = self.data_list[index].split('/')[-1]

        if self.training and hasattr(self, 'caption_cfg'):
            if self.caption_cfg.get('CAPTION_CORR_PATH_IN_ONE_FILE', True):
                image_corr_dict, image_name_dict = self.get_caption_image_corr_and_name_from_memory(scene_name, index)
            else:
                image_corr_dict, image_name_dict = self.get_caption_image_corr_and_name_from_file(scene_name)
            caption_data = self.select_caption_and_idx_all(scene_name, image_name_dict, image_corr_dict)
        else:
            caption_data = None

        # subsample
        subsample_idx = self.subsample(xyz_all, label_all, self.downsampling_scale)
        xyz, rgb, label, inst_label, binary_label, origin_idx = self.filter_by_index(
            [xyz_all, rgb_all, label_all, inst_label_all, binary_label_all, origin_idx], subsample_idx
        )

        data_dict = {
            'points_xyz': xyz, 'rgb': rgb, 'labels': label, 'inst_label': inst_label,
            'binary_labels': binary_label, 'origin_idx': origin_idx, 'pc_count': pc_count,
            'caption_data': caption_data, 'ids': index, 'scene_name': scene_name
        }

        if self.training:
            # perform augmentations
            data_dict = self.augmentor.forward(data_dict)
            if not data_dict['valid']:
                return self.__getitem__(np.random.randint(self.__len__()))
        else:
            xyz_voxel_scale = xyz * self.voxel_scale
            xyz_voxel_scale -= xyz_voxel_scale.min(0)
            data_dict['points_xyz_voxel_scale'] = xyz_voxel_scale
            data_dict['points'] = xyz

        # prepare features for voxelization
        if self.dataset_cfg.DATA_PROCESSOR.rgb_as_feat:
            data_dict['feats'] = data_dict['rgb']

        if self.dataset_cfg.DATA_PROCESSOR.xyz_as_feat:
            if 'feats' in data_dict:
                data_dict['feats'] = np.concatenate((data_dict['feats'], data_dict['points_xyz']), axis=1)
            else:
                data_dict['feats'] = data_dict['points_xyz']

        data_dict = self.data_processor.forward(data_dict)

        # data_dict.pop('points_xyz')
        data_dict.pop('rgb')
        return data_dict

    def get_test_item(self, item):
        index = item % len(self.data_list)
        xyz_all, rgb_all, label_all, inst_label_all, binary_label_all = \
            self.load_data(index)
        xyz_all -= xyz_all.mean(0)
        rgb_all = rgb_all / 127.5 - 1.0
        inds = np.arange(xyz_all.shape[0])
        piece_1 = inds[::4]
        piece_2 = inds[1::4]
        piece_3 = inds[2::4]
        piece_4 = inds[3::4]
        xyz_list = []
        xyz_middle_list = []
        rgb_list = []
        # semantic_label_list = []
        # instance_label_list = []
        # binary_label_list = []
        for batch, piece in enumerate([piece_1, piece_2, piece_3, piece_4]):
            xyz_middle = xyz_all[piece]
            xyz = xyz_middle * self.voxel_scale
            xyz -= xyz.min(0)
            xyz_list.append(np.concatenate([np.full((xyz.shape[0], 1), batch), xyz], 1))
            xyz_middle_list.append(xyz_middle)
            rgb_list.append(rgb_all[piece])
            # semantic_label_list.append(label_all[piece])
            # instance_label_list.append(inst_label_all[piece])
            # binary_label_list.append(binary_label_all[piece])
        xyz = np.concatenate(xyz_list, 0)
        xyz_middle = np.concatenate(xyz_middle_list, 0)
        rgb = np.concatenate(rgb_list, 0)
        # semantic_label = np.concatenate(semantic_label_list, 0)
        # instance_label = np.concatenate(instance_label_list, 0)
        # binary_label = np.concatenate(binary_label_list, 0)
        # valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        # instance_label = self.get_valid_inst_label(instance_label, (semantic_label != self.ignore))  # TODO remove this
        data_dict = {'points_xyz': xyz_middle, 'points_xyz_voxel_scale': xyz, 'rgb': rgb,
                     'labels': label_all, 'inst_label': inst_label_all, 'ids': index,
                     'scene_name': self.data_list[index].split('/')[-1][:-4]}

        if self.dataset_cfg.DATA_PROCESSOR.rgb_as_feat:
            data_dict['feats'] = data_dict['rgb']

        if self.dataset_cfg.DATA_PROCESSOR.xyz_as_feat:
            if 'feats' in data_dict:
                data_dict['feats'] = np.concatenate((data_dict['feats'], data_dict['points_xyz']), axis=1)
            else:
                data_dict['feats'] = data_dict['points_xyz']
        # data_dict.pop('points_xyz')
        data_dict.pop('rgb')
        return data_dict

    def _del__(self):
        if not self.cache:
            return

        for item in self.data_list:
            if os.path.exists("/dev/shm/{}_nn".format(item)):
                sa_delete("shm://{}_nn".format(item))


class S3DISInstDataset(S3DISDataset):
    def __init__(self, dataset_cfg, class_names, training, root_path, logger=None):
        S3DISDataset.__init__(self, dataset_cfg, class_names, training, root_path, logger=logger)
        self.inst_class_idx = dataset_cfg.inst_class_idx
        self.inst_label_shift = dataset_cfg.inst_label_shift
        if 'base_class_idx' in dataset_cfg:
            self.base_inst_class_idx = self.base_class_idx
            self.novel_inst_class_idx = self.novel_class_idx
        self.sem2ins_classes = dataset_cfg.sem2ins_classes

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)

        # get instance infos
        label, inst_label, binary_label = data_dict['labels'], data_dict['inst_label'], data_dict['binary_labels']
        points = data_dict['points_xyz']
        if self.training:
            inst_label[binary_label == 0] = self.ignore_label
        inst_label = self.get_valid_inst_label(inst_label, label != self.ignore_label)
        if self.training and inst_label.max() < 0:
            return self.__getitem__(np.random.randint(self.__len__()))
        info = self.get_inst_info(points, inst_label.astype(np.int32), label)
        data_dict['inst_label'] = inst_label
        data_dict.update(info)
        return data_dict

    def get_test_item(self, item):
        index = item % len(self.data_list)
        xyz_all, rgb_all, label_all, inst_label_all, binary_label_all = \
            self.load_data(index)
        xyz_all -= xyz_all.mean(0)
        rgb_all = rgb_all / 127.5 - 1.0
        inds = np.arange(xyz_all.shape[0])
        piece_1 = inds[::4]
        piece_2 = inds[1::4]
        piece_3 = inds[2::4]
        piece_4 = inds[3::4]
        xyz_list = []
        xyz_middle_list = []
        rgb_list = []
        semantic_label_list = []
        instance_label_list = []
        binary_label_list = []
        for batch, piece in enumerate([piece_1, piece_2, piece_3, piece_4]):
            xyz_middle = xyz_all[piece]
            xyz = xyz_middle * self.voxel_scale
            xyz -= xyz.min(0)
            xyz_list.append(np.concatenate([np.full((xyz.shape[0], 1), batch), xyz], 1))
            xyz_middle_list.append(xyz_middle)
            rgb_list.append(rgb_all[piece])
            semantic_label_list.append(label_all[piece])
            instance_label_list.append(inst_label_all[piece])
            binary_label_list.append(binary_label_all[piece])
        xyz = np.concatenate(xyz_list, 0)
        xyz_middle = np.concatenate(xyz_middle_list, 0)
        rgb = np.concatenate(rgb_list, 0)
        semantic_label = np.concatenate(semantic_label_list, 0)
        instance_label = np.concatenate(instance_label_list, 0)
        binary_label = np.concatenate(binary_label_list, 0)
        # valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        # instance_label = self.get_valid_inst_label(instance_label, (semantic_label != self.ignore))  # TODO remove this

        data_dict = {'points_xyz': xyz_middle, 'points_xyz_voxel_scale': xyz, 'rgb': rgb,
                     'labels': semantic_label, 'inst_label': instance_label, 'binary_labels': binary_label,
                     'ids': index, 'scene_name': self.data_list[index].split('/')[-1][:-4]}

        if self.dataset_cfg.DATA_PROCESSOR.rgb_as_feat:
            data_dict['feats'] = data_dict['rgb']

        if self.dataset_cfg.DATA_PROCESSOR.xyz_as_feat:
            if 'feats' in data_dict:
                data_dict['feats'] = np.concatenate((data_dict['feats'], data_dict['points_xyz']), axis=1)
            else:
                data_dict['feats'] = data_dict['points_xyz']
        # data_dict.pop('points_xyz')
        # data_dict.pop('rgb')
        return data_dict

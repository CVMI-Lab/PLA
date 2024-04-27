import SharedArray as SA
import os
import numpy as np
import glob
import torch
import pickle
import json
import cv2
import copy
from pathlib import Path

from torch.utils import data

from ..indoor_dataset import IndoorDataset
from ...utils.common_utils import sa_create, sa_delete


class STPLS3DDataset(IndoorDataset):
    def __init__(self, dataset_cfg, class_names, training, root_path, logger=None, split=None):
        super(STPLS3DDataset, self).__init__(
            dataset_cfg, class_names, training, root_path, logger=logger, split=split
        )

        self.data_suffix = dataset_cfg.DATA_SPLIT.data_suffix
        self.data_list = sorted(
            glob.glob(str(self.root_path / dataset_cfg.DATA_SPLIT[self.mode]) + '/*' + self.data_suffix))
        self.split_file = dataset_cfg.DATA_SPLIT[self.mode]
        self.need_super_voxel = self.dataset_cfg.get('NEED_SV', False)
        self.put_data_to_shm()

        if self.training and hasattr(self, 'caption_cfg') and self.caption_cfg.get('CAPTION_CORR_PATH_IN_ONE_FILE', True):
            self.scene_image_corr_infos, self.scene_image_corr_entity_infos = self.include_caption_infos()

        self.load_image = self.dataset_cfg.get('LOAD_IMAGE', None)
        self.depth_image_scale = self.dataset_cfg.get('DEPTH_IMAGE_SCALE', None)
        self.image_path = self.dataset_cfg.get('IMAGE_PATH', None)
        self.pred_2d_path = self.dataset_cfg.get('PRED_2D_PATH', None)

        if self.load_image:
            self.pc_means = json.load(open(str(self.root_path / 'pc_mean.json')))

        self.logger.info(
            "Totally {} samples in {} set.".format(
                len(self.data_list) * (self.repeat if self.training else 1), self.mode))

    def __len__(self):
        length = len(self.data_list) * (self.repeat if self.training else 1)

        if self._merge_all_iters_to_one_epoch:
            return length * self.total_epochs
        else:
            return length

    def put_data_to_shm(self):
        for item in self.data_list:
            if self.cache and not os.path.exists("/dev/shm/stpls3d_{}".format(item.split('/')[-1][:-len(self.data_suffix)] + '_xyz_')):
                xyz, rgb, label, inst_label, *others = torch.load(item)
                sa_create("shm://stpls3d_{}".format(item.split('/')[-1][:-len(self.data_suffix)] + '_xyz_'), xyz)
                sa_create("shm://stpls3d_{}".format(item.split('/')[-1][:-len(self.data_suffix)] + '_rgb_'), rgb)
                sa_create("shm://stpls3d_{}".format(item.split('/')[-1][:-len(self.data_suffix)] + '_label_'), label)
                sa_create("shm://stpls3d_{}".format(item.split('/')[-1][:-len(self.data_suffix)] + '_inst_label_'), inst_label)
                if self.need_super_voxel:
                    sv = others[1]
                    sa_create("shm://stpls3d_{}".format(item.split('/')[-1][:-len(self.data_suffix)] + '_sv_'), np.array(sv))

    def load_data(self, index):
        fn = self.data_list[index]
        if self.cache:
            xyz = SA.attach("shm://stpls3d_{}".format(fn.split('/')[-1][:-len(self.data_suffix)] + '_xyz_')).copy()
            rgb = SA.attach("shm://stpls3d_{}".format(fn.split('/')[-1][:-len(self.data_suffix)] + '_rgb_')).copy()
            if self.split_file.find('test') < 0:
                label = SA.attach("shm://stpls3d_{}".format(fn.split('/')[-1][:-len(self.data_suffix)] + '_label_')).copy()
                inst_label = SA.attach("shm://stpls3d_{}".format(fn.split('/')[-1][:-len(self.data_suffix)] + '_inst_label_')).copy()
            else:
                label = np.full(xyz.shape[0], self.ignore_label).astype(np.int64)
                inst_label = np.full(xyz.shape[0], self.ignore_label).astype(np.int64)
            if self.need_super_voxel:
                sv = SA.attach("shm://stpls3d_{}".format(fn.split('/')[-1][:-4] + '_sv_')).copy()
        else:
            xyz, rgb, label, inst_label, *others = torch.load(fn)
            if self.need_super_voxel:
                sv = others[1]

        # base / novel label
        if hasattr(self, 'base_class_mapper'):
            binary_label = self.binary_class_mapper[label.astype(np.int64)].astype(np.float32)
        else:
            binary_label = np.ones_like(label)
        if self.class_mode == 'base':
            label = self.base_class_mapper[label.astype(np.int64)]
        elif self.class_mode == 'all' and hasattr(self, 'ignore_class_idx'):
            label = self.valid_class_mapper[label.astype(np.int64)]
        inst_label[label == self.ignore_label] = self.ignore_label

        if self.need_super_voxel:
            return xyz, rgb, label, inst_label, binary_label, sv
        else:
            return xyz, rgb, label, inst_label, binary_label

    def __getitem__(self, item):
        index = item % len(self.data_list)
        xyz, rgb, label, inst_label, binary_label, *others = self.load_data(index)

        pc_count = xyz.shape[0]
        origin_idx = np.arange(xyz.shape[0]).astype(np.int64)
        # === caption ===
        scene_name = self.data_list[index].split('/')[-1].split('.')[0]

        # get captioning data
        if self.training and hasattr(self, 'caption_cfg'):
            if self.caption_cfg.get('CAPTION_CORR_PATH_IN_ONE_FILE', True):
                image_corr_dict, image_name_dict = self.get_caption_image_corr_and_name_from_memory(scene_name, index)
            else:
                image_corr_dict, image_name_dict = self.get_caption_image_corr_and_name_from_file(scene_name)
            caption_data = self.select_caption_and_idx_all(scene_name, image_name_dict, image_corr_dict)
        else:
            caption_data = None

        if not self.rgb_norm:
            rgb = (rgb + 1) * 127.5

        data_dict = {
            'points_xyz': xyz, 'rgb': rgb, 'labels': label, 'inst_label': inst_label,
            'binary_labels': binary_label, 'origin_idx': origin_idx, 'pc_count': pc_count,
            'caption_data': caption_data, 'ids': index, 'scene_name': scene_name
        }

        if self.need_super_voxel:
            sv = others[0]
            sv = self.get_valid_inst_label(sv)
            data_dict['super_voxel'] = sv

        # === instance pseudo offset label ====
        if self.training and hasattr(self, 'pseudo_label_dir'):
            # print(self.pseudo_label_dir)
            index = item % len(self.data_list)
            fn = self.data_list[index]
            pseudo_offset = self.load_pseudo_labels(fn.split('/')[-1][:-4], dtype=np.float, shape=(-1, 3))
            data_dict['pt_offset_mask'] = (pseudo_offset == 0).sum(1) != 3
            # pseudo_offset[(pseudo_offset == 0).sum(1) == 3] = -100.
            data_dict['pseudo_offset_target'] = xyz + pseudo_offset

        # === need super voxels ===
        if self.need_super_voxel:
            sv = others[0]
            sv = self.get_valid_inst_label(sv)
            data_dict['super_voxel'] = sv

        # === load images ===
        if self.load_image:
            info = {'scene_name': scene_name, 'depth_image_size': self.depth_image_scale}
            data_dict = self.get_image(info, data_dict)

        # get kd data
        # KD label will only carrys on 3D zero-shot
        if self.load_kd_label:
            kd_labels, kd_mask = self.get_kd_data(scene_name)
            if self.training:
                data_dict['kd_labels'] = kd_labels
                data_dict['kd_labels_mask'] = kd_mask
            else:
                data_dict['adapter_feats'] = kd_labels
                data_dict['adapter_feats_mask'] = kd_mask

        if self.training:
            # perform augmentations
            data_dict = self.augmentor.forward(data_dict)
            if not data_dict['valid']:
                return STPLS3DDataset.__getitem__(np.random.randint(self.__len__()))
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

        # visualization debug code
        # import tools.visual_utils.open3d_vis_utils as vis
        # vis_dict = {
        #     'points': data_dict['points'],
        #     'point_colors': data_dict['rgb'],
        #     'point_size': 2.0
        # }
        # vis.dump_vis_dict(vis_dict)
        # import ipdb;
        # ipdb.set_trace(context=20)

        # data_dict.pop('points_xyz')
        return data_dict

    def __del__(self):
        if not self.cache:
            return

        for item in self.data_list:
            if os.path.exists("/dev/shm/stpls3d_{}".format(item.split('/')[-1][:-4] + '_xyz_')):
                sa_delete("shm://stpls3d_{}".format(item.split('/')[-1][:-4] + '_rgb_'))
                sa_delete("shm://stpls3d_{}".format(item.split('/')[-1][:-4] + '_xyz_'))
                sa_delete("shm://{}".format(item.split('/')[-1][:-4] + '_label_'))
                sa_delete("shm://{}".format(item.split('/')[-1][:-4] + '_inst_label_'))


class STPLS3DInstDataset(STPLS3DDataset):
    def __init__(self, dataset_cfg, class_names, training, root_path, logger=None, split=None):
        STPLS3DDataset.__init__(self, dataset_cfg, class_names, training, root_path, logger=logger, split=split)
        self.inst_class_idx = dataset_cfg.inst_class_idx
        self.inst_label_shift = dataset_cfg.inst_label_shift
        if 'base_class_idx' in dataset_cfg:
            self.base_inst_class_idx = np.array(self.base_class_idx)[dataset_cfg.inst_label_shift:] - self.inst_label_shift
            self.novel_inst_class_idx = np.array(self.novel_class_idx) - self.inst_label_shift
        self.sem2ins_classes = dataset_cfg.sem2ins_classes
        self.NYU_ID = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)
        # get instance infos
        # info = self.get_instance_info(xyz_mid, inst_label.astype(np.int32), label)
        label, inst_label, binary_label = data_dict['labels'], data_dict['inst_label'], data_dict['binary_labels']
        points = data_dict['points_xyz']
        if self.training:
            inst_label[binary_label == 0] = self.ignore_label
        inst_label = self.get_valid_inst_label(inst_label, label != self.ignore_label)
        if self.training and inst_label.max() < 0:
            return STPLS3DInstDataset.__getitem__(np.random.randint(self.__len__()))
        info = self.get_inst_info(points, inst_label.astype(np.int32), label)
        if self.training and hasattr(self, 'pseudo_label_dir'):
            # print('update pseudo label')
            info['pt_offset_label'][binary_label == 0] = (data_dict['pseudo_offset_target'] - points)[binary_label == 0]
            data_dict['pt_offset_mask'] = (data_dict['pt_offset_mask'] & (binary_label == 0)) | (inst_label != self.ignore_label)
            del data_dict['pseudo_offset_target']
        data_dict['inst_label'] = inst_label
        data_dict.update(info)
        return data_dict

    def get_inst_info(self, xyz, instance_label, semantic_label):
        ret = super().get_inst_info(xyz, instance_label, semantic_label)
        ret['inst_cls'] = [x - self.inst_label_shift if x != -100 else x for x in ret['inst_cls']]
        return ret

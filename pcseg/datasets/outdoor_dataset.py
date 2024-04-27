import torch
import numpy as np
from collections import defaultdict
from .dataset import DatasetTemplate
from .augmentor.data_augmentor import DataAugmentor

from .processor.data_processor import DataProcessor


class OutdoorDataset(DatasetTemplate):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None, split=None):
        super(OutdoorDataset, self).__init__(dataset_cfg, class_names, training, root_path, logger=logger, split=split)

        self.voxel_size = self.dataset_cfg.VOXEL_SIZE

        self.augmentor = DataAugmentor(self.dataset_cfg)

        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range, training=training,
            num_point_features=dataset_cfg.NUM_POINT_FEATURES
        )

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    # instance seg
    def get_valid_inst_label(self, inst_label, valid_mask=None):
        if valid_mask is not None:
            inst_label[~valid_mask] = self.ignore_label
        label_set = np.unique(inst_label[inst_label != self.ignore_label])
        if len(label_set) > 0:
            remapper = np.full((int(label_set.max()) + 1,), self.ignore_label)
            remapper[label_set.astype(np.int64)] = np.arange(len(label_set))
            inst_label[inst_label != self.ignore_label] = \
                remapper[inst_label[inst_label != self.ignore_label].astype(np.int64)]
        # inst_label[~valid_mask] = self.ignore_label
        # j = 0
        # while (j < inst_label.max()):
        #     if (len(np.where(inst_label == j)[0]) == 0):
        #         inst_label[inst_label == inst_label.max()] = j
        #     j += 1
        return inst_label 

    def filter_instance_with_min_points(self, inst_label, min_gt_pts):
        num_pts = np.bincount(inst_label[inst_label != self.ignore_label])
        ignore_inst_id = (num_pts < min_gt_pts).nonzero()[0]
        inst_label_mask = np.ones(inst_label.shape, dtype=bool)
        for id in ignore_inst_id:
            inst_label_mask[inst_label == id] = False
        return inst_label_mask

    def get_inst_info(self, xyz, inst_label, semantic_label):

        inst_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * self.ignore_label 
        # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        inst_pointnum = []   # (nInst), int
        inst_cls = []
        if len(inst_label[inst_label != self.ignore_label]) == 0:
            inst_num = 0
        else:
            inst_num = int(inst_label[inst_label != self.ignore_label].max()) + 1
        for i_ in range(inst_num):
            inst_idx_i = np.where(inst_label == i_)

            ### inst_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            inst_info_i = inst_info[inst_idx_i]
            inst_info_i[:, 0:3] = mean_xyz_i
            inst_info_i[:, 3:6] = min_xyz_i
            inst_info_i[:, 6:9] = max_xyz_i
            inst_info[inst_idx_i] = inst_info_i

            ### inst_pointnum
            inst_pointnum.append(inst_idx_i[0].size)

            ### inst cls
            cls_idx = inst_idx_i[0][0]
            inst_cls.append(semantic_label[cls_idx])
        pt_offset_label = inst_info[:, 0:3] - xyz
        return {'inst_num': inst_num, 'inst_pointnum': inst_pointnum, \
            'inst_cls': inst_cls, 'pt_offset_label': pt_offset_label}
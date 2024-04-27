import SharedArray as SA
import os
import numpy as np
import glob
import torch
import pickle
import json
import cv2
import copy
import yaml

from ..indoor_dataset import IndoorDataset
from ...utils.common_utils import check_exists
from .calibration_kitti import read_calib


class KittiDataset(IndoorDataset):
    def __init__(self, dataset_cfg, class_names, training, root_path, logger=None, split=None):
        super(KittiDataset, self).__init__(
            dataset_cfg, class_names, training, root_path, logger=logger, split=split
        )

        with open(os.path.join(dataset_cfg.INFO_PATH, 'semantic-kitti.yaml'), 'r') as fin:
            semkittiyaml = yaml.safe_load(fin)
        self.split = semkittiyaml['split'][dataset_cfg.DATA_SPLIT[self.mode]]
        self.learning_map = semkittiyaml['learning_map']
        self.learning_map_inv = semkittiyaml['learning_map_inv']
        for k, v in self.learning_map.items():
            if v == 0:
                new_v = self.ignore_label
            elif v < 9:
                new_v = v + 10
            else:
                new_v = v - 9
            self.learning_map[k] = new_v

        self.data_suffix = dataset_cfg.DATA_SPLIT.data_suffix
        self.data_list = self.get_filenames()
        self.filename_list = ['_'.join(d.split('/')[-4:]) for d in self.data_list]
        self.with_label = True
        # self.split_file = dataset_cfg.DATA_SPLIT[self.mode]
        self.need_super_voxel = self.dataset_cfg.get('NEED_SV', False)
        # self.put_data_to_shm()

        if self.training and hasattr(self, 'caption_cfg') and self.caption_cfg.get('CAPTION_CORR_PATH_IN_ONE_FILE', True):
            self.scene_image_corr_infos, self.scene_image_corr_entity_infos = self.include_caption_infos()

        self.load_image = self.dataset_cfg.get('LOAD_IMAGE', None)
        self.depth_image_scale = self.dataset_cfg.get('DEPTH_IMAGE_SCALE', None)
        self.image_path = self.dataset_cfg.get('IMAGE_PATH', None)
        self.pred_2d_path = self.dataset_cfg.get('PRED_2D_PATH', None)

        if self.dataset_cfg.get('FOV_POINTS_ONLY', False) or self.load_image:
            self.img_shape_dict = pickle.load(open(self.root_path / self.dataset_cfg.IMG_SHAPE_PATH, 'rb'))

        self.logger.info(
            "Totally {} samples in {} set.".format(
                len(self.data_list) * (self.repeat if self.training else 1), self.mode))

    def get_filenames(self):
        with open(os.path.join(self.dataset_cfg.DATA_SPLIT['root'], f'kitti_{self.dataset_cfg.DATA_SPLIT[self.mode]}.txt'), 'r') as fin:
            filenames = fin.readlines()
        if self.oss_client:
            filenames = [os.path.join(self.oss_root_path, f.strip()) for f in filenames]
        else:
            filenames = [str(self.root_path / f.strip()) for f in filenames]

        filenames = sorted(filenames * self.repeat)
        return filenames

    def __len__(self):
        length = len(self.data_list) * (self.repeat if self.training else 1)

        if self._merge_all_iters_to_one_epoch:
            return length * self.total_epochs
        else:
            return length

    def get_calib(self, seq_idx):
        if self.oss_client:
            calib_file = os.path.join(self.root_path, 'dataset/sequences', seq_idx, 'calib.txt')
        else:
            calib_file = self.root_path / 'dataset/sequences' / seq_idx / 'calib.txt'
        assert check_exists(calib_file)
        return read_calib(calib_file)

    def get_fov_flag(self, xyz, sequence, index, return_img_points=False):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        calib = self.get_calib(sequence)
        proj_matrix = np.matmul(calib["P2"], calib["Tr"])

        fov_flag = xyz[:, 0] > 0  # only keep point in front of the vehicle
        xyz_hcoords = np.concatenate([xyz[fov_flag], np.ones([fov_flag.sum(), 1], dtype=np.float32)], axis=1)
        img_points = (proj_matrix @ xyz_hcoords.T).T

        img_shape = self.img_shape_dict[self.filename_list[index][:-4].replace('velodyne', 'image_2')]

        pts_img = img_points[..., :2] / np.expand_dims(img_points[:, 2], axis=1)
        pts_rect_depth = img_points[..., 2]

        # pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        fov_flag[fov_flag] = pts_valid_flag

        if return_img_points:
            return fov_flag, pts_img[pts_valid_flag]
        else:
            return fov_flag

    def get_data_list(self):
        return self.filename_list

    def load_data(self, index):
        filename = self.data_list[index]
        sequence = filename.split('/')[-3]
        if self.oss_client:
            data = np.frombuffer(self.oss_client.get(filename), dtype=np.float32).reshape(-1, 4)
        else:
            data = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        xyz, rgb = data[:, :3], data[:, 3:]
        if self.with_label:
            label = np.fromfile(
                filename.replace('velodyne', 'labels').replace('bin', 'label'), dtype=np.int32)
            semantic_label = label & 0xFFFF
            semantic_label = np.vectorize(self.learning_map.__getitem__)(semantic_label)
            stuff_inds = semantic_label <= 10
            instance_label = label
            instance_label[stuff_inds] = self.ignore_label
            assert semantic_label.shape[0] == xyz.shape[0]
        else:
            semantic_label = np.zeros(xyz.shape[0])
            instance_label = np.zeros(xyz.shape[0])

        # if FOV only
        if self.dataset_cfg.get('FOV_POINTS_ONLY', False):
            fov_flag = self.get_fov_flag(xyz, sequence, index)
            xyz = xyz[fov_flag]
            rgb = rgb[fov_flag]
            semantic_label = semantic_label[fov_flag]
            instance_label = instance_label[fov_flag]
            # === debug only ===
            # import open3d as o3d
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz)
            # pcd.colors = o3d.utility.Vector3dVector(rgb)
            # o3d.io.write_point_cloud(save_path, pcd)


        # base / novel label
        if hasattr(self, 'base_class_mapper'):
            binary_label = self.binary_class_mapper[semantic_label.astype(np.int64)].astype(np.float32)
        else:
            binary_label = np.ones_like(semantic_label)
        if self.class_mode == 'base':
            semantic_label = self.base_class_mapper[semantic_label.astype(np.int64)]
        elif self.class_mode == 'all' and hasattr(self, 'ignore_class_idx'):
            semantic_label = self.valid_class_mapper[semantic_label.astype(np.int64)]
        instance_label[semantic_label == self.ignore_label] = self.ignore_label

        return xyz, rgb, semantic_label, instance_label, binary_label

    def __getitem__(self, item):
        index = item % len(self.data_list)
        xyz, rgb, label, inst_label, binary_label, *others = self.load_data(index)

        pc_count = xyz.shape[0]
        origin_idx = np.arange(xyz.shape[0]).astype(np.int64)
        # === caption ===
        scene_name = self.data_list[index]

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
            fn = self.filename_list[index]
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
                return KittiDataset.__getitem__(self, np.random.randint(self.__len__()))
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

    def get_image(self, info, data_dict):
        data_dict['point_img_1d'] = {}
        data_dict['point_img'] = {}
        data_dict['point_img_idx'] = {}
        data_dict['image_shape'] = {}

        filename = data_dict['scene_name']
        sequence = filename.split('_')[1]
        img_shape = self.img_shape_dict[filename.replace('velodyne', 'image_2')]

        fov_flag, point_img = self.get_fov_flag(data_dict['points_xyz'], sequence, info['idx'], return_img_points=True)
        point_img_idx = fov_flag.nonzero()[0]

        data_dict['point_img']['0'] = point_img
        data_dict['point_img_idx']['0'] = point_img_idx
        data_dict['image_shape']['0'] = img_shape

        data_dict['depth_image_size'] = img_shape
        return data_dict

    @staticmethod
    def project_point_to_image(points_world, pose_path, depth_path, color_path, image_size, depth_intrinsic):
        fx = depth_intrinsic[0, 0]
        fy = depth_intrinsic[1, 1]
        cx = depth_intrinsic[0, 2]
        cy = depth_intrinsic[1, 2]
        bx = depth_intrinsic[0, 3]
        by = depth_intrinsic[1, 3]

        # == processing depth ===
        depth_img = cv2.imread(depth_path, -1)  # read 16bit grayscale image
        depth_shift = 1000.0
        depth = depth_img / depth_shift
        depth_mask = (depth_img != 0)

        # == processing color ===
        color_image = cv2.imread(color_path)
        color_image_shape = color_image.shape
        color_image = cv2.resize(color_image, (image_size[1], image_size[0]))
        # color_image = np.reshape(color_image[mask], [-1,3])  ##########
        color_image = np.reshape(color_image, [-1, 3])
        colors = np.zeros_like(color_image)
        colors[:, 0] = color_image[:, 2]
        colors[:, 1] = color_image[:, 1]
        colors[:, 2] = color_image[:, 0]

        # == processing pose ===
        pose = np.loadtxt(pose_path)

        # == 3D to camera coordination ===
        points = np.hstack((points_world[..., :3], np.ones((points_world.shape[0], 1))))
        points = np.dot(points, np.linalg.inv(np.transpose(pose)))

        # == camera to image coordination ===
        u = (points[..., 0] - bx) * fx / points[..., 2] + cx
        v = (points[..., 1] - by) * fy / points[..., 2] + cy
        d = points[..., 2]
        u = (u + 0.5).astype(np.int32)
        v = (v + 0.5).astype(np.int32)

        # debug test
        # img = np.zeros((image_size[0], image_size[1], 3))
        # for jj in range(image_size[0]):
        #     for kk in range(image_size[1]):
        #         curr_d = d[(u == kk) & (v == jj) & (d >= 0)]
        #         if curr_d.shape[0] > 0:
        #             idx = curr_d.argmin()
        #             img[jj, kk] = points_world[..., 3:6][..., ::-1][(u == kk) & (v == jj) & (d >= 0)][idx]
        #         else:
        #             img[jj, kk] = 255.0
        # cv2.imwrite('temp.png', img)
        # import ipdb; ipdb.set_trace()

        # filter out invalid points
        point_valid_mask = (d >= 0) & (u < image_size[1]) & (v < image_size[0]) & (u >= 0) & (v >= 0)
        point_valid_idx = np.where(point_valid_mask)[0]
        point2image_coords = v * image_size[1] + u
        valid_point2image_coords = point2image_coords[point_valid_idx]

        depth = depth.reshape(-1)
        depth_mask = depth_mask.reshape(-1)
        # u_, v_ = np.meshgrid(np.linspace(0, image_size[1] - 1, image_size[1]), np.linspace(0, image_size[0] - 1, image_size[0]))
        # image_coords = (v_ * image_size[1] + u_).reshape(-1)
        image_depth = depth[valid_point2image_coords.astype(np.int64)]
        depth_mask = depth_mask[valid_point2image_coords.astype(np.int64)]
        point2image_depth = d[point_valid_idx]
        depth_valid_mask = depth_mask & (np.abs(image_depth - point2image_depth) <= 0.2 * image_depth)
        # depth_valid_idx = np.where(depth_valid_mask)[0]  # corresponding image coords
        point2image_coords_1d = valid_point2image_coords[depth_valid_mask]  # corresponding image coords
        point2image_coords_u = point2image_coords_1d % image_size[1]  # (width, long)
        point2image_coords_v = point2image_coords_1d // image_size[1]  # (height, short)
        point2image_coords_2d = np.concatenate([point2image_coords_u[:, None], point2image_coords_v[:, None]], axis=-1)
        point_valid_idx = point_valid_idx[depth_valid_mask]  # corresponding point idx

        return point_valid_idx, point2image_coords_1d, point2image_coords_2d, color_image_shape


class KittiPanopticDataset(KittiDataset):
    def __init__(self, dataset_cfg, class_names, training, root_path, logger=None, split=None):
        KittiDataset.__init__(self, dataset_cfg, class_names, training, root_path, logger=logger, split=split)
        self.inst_class_idx = dataset_cfg.inst_class_idx
        self.stuff_class_idx = dataset_cfg.stuff_class_idx
        self.inst_label_shift = dataset_cfg.inst_label_shift
        if 'base_class_idx' in dataset_cfg:
            # panoptic seg, stuff first
            base_inst_class_idx = np.array(list(set(self.base_class_idx[dataset_cfg.inst_label_shift:]) & set(self.inst_class_idx)))
            novel_inst_class_idx = np.array(list(set(self.novel_class_idx[dataset_cfg.inst_label_shift:]) & set(self.inst_class_idx)))
            self.base_inst_class_idx = base_inst_class_idx - self.inst_label_shift
            self.novel_inst_class_idx = novel_inst_class_idx - self.inst_label_shift
        self.sem2ins_classes = dataset_cfg.sem2ins_classes
        # self.NYU_ID = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

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
            return KittiPanopticDataset.__getitem__(self, np.random.randint(self.__len__()))
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
        ret['inst_cls'] = [x - self.inst_label_shift if x != self.ignore_label else x for x in ret['inst_cls']]
        return ret

    def get_valid_inst_label(self, instance_label, valid_idxs):
        instance_label[~valid_idxs] = self.ignore_label
        # instance_label = instance_label[valid_idxs]
        ins_label_map = {}
        new_id = 0
        instance_ids = np.unique(instance_label)
        for id in instance_ids:
            if id == self.ignore_label:
                ins_label_map[id] = id
                continue
            ins_label_map[id] = new_id
            new_id += 1
        instance_label = np.vectorize(ins_label_map.__getitem__)(instance_label)
        return instance_label

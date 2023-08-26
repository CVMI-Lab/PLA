import SharedArray as SA
import os
import numpy as np
import glob
import torch
import pickle
import json
import cv2
import copy

from torch.utils import data

from ..indoor_dataset import IndoorDataset
from ...utils.common_utils import sa_create, sa_delete


class ScanNetDataset(IndoorDataset):
    def __init__(self, dataset_cfg, class_names, training, root_path, logger=None):
        super(ScanNetDataset, self).__init__(
            dataset_cfg, class_names, training, root_path, logger=logger
        )

        self.data_suffix = dataset_cfg.DATA_SPLIT.data_suffix
        self.data_list = sorted(
            glob.glob(str(self.root_path / dataset_cfg.DATA_SPLIT[self.mode]) + '/*' + self.data_suffix))
        self.split_file = dataset_cfg.DATA_SPLIT[self.mode]
        self.put_data_to_shm()

        if self.training and hasattr(self, 'caption_cfg') and self.caption_cfg.get('CAPTION_CORR_PATH_IN_ONE_FILE', True):
            self.scene_image_corr_infos, self.scene_image_corr_entity_infos = self.include_caption_infos()

        self.load_image = self.dataset_cfg.get('LOAD_IMAGE', None)
        self.depth_image_scale = self.dataset_cfg.get('DEPTH_IMAGE_SCALE', None)
        self.image_path = self.dataset_cfg.get('IMAGE_PATH', None)

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
        n_classes = self.n_classes
        for item in self.data_list:
            if self.cache and not os.path.exists("/dev/shm/scannet_{}".format(item.split('/')[-1][:-4] + '_xyz_{}'.format(n_classes))):
                if self.split_file.find('test') < 0:
                    xyz, rgb, label, inst_label, *others = torch.load(item)
                    sa_create("shm://scannet_{}".format(item.split('/')[-1][:-4] + '_label_{}'.format(n_classes)), label)
                    sa_create("shm://scannet_{}".format(item.split('/')[-1][:-4] + '_inst_label_{}'.format(n_classes)), inst_label)
                else:
                    xyz, rgb = torch.load(item)
                sa_create("shm://scannet_{}".format(item.split('/')[-1][:-4] + '_xyz_{}'.format(n_classes)), xyz)
                sa_create("shm://scannet_{}".format(item.split('/')[-1][:-4] + '_rgb_{}'.format(n_classes)), rgb)

    def load_data(self, index):
        n_classes = self.n_classes
        fn = self.data_list[index]
        if self.cache:
            xyz = SA.attach("shm://scannet_{}".format(fn.split('/')[-1][:-4] + '_xyz_{}'.format(n_classes))).copy()
            rgb = SA.attach("shm://scannet_{}".format(fn.split('/')[-1][:-4] + '_rgb_{}'.format(n_classes))).copy()
            if self.split_file.find('test') < 0:
                label = SA.attach("shm://scannet_{}".format(fn.split('/')[-1][:-4] + '_label_{}'.format(n_classes))).copy()
                inst_label = SA.attach("shm://scannet_{}".format(fn.split('/')[-1][:-4] + '_inst_label_{}'.format(n_classes))).copy()
            else:
                label = np.full(xyz.shape[0], self.ignore_label).astype(np.int64)
                inst_label = np.full(xyz.shape[0], self.ignore_label).astype(np.int64)
        else:
            if self.split_file.find('test') < 0:
                xyz, rgb, label, inst_label, *others = torch.load(fn)
            else:
                xyz, rgb = torch.load(fn)
                label = np.full(xyz.shape[0], self.ignore_label)
                inst_label = np.full(xyz.shape[0], self.ignore_label)

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

        return xyz, rgb, label, inst_label, binary_label

    def __getitem__(self, item):
        index = item % len(self.data_list)
        xyz, rgb, label, inst_label, binary_label = self.load_data(index)

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

        # === load images ===
        if self.load_image:
            info = {'scene_name': scene_name, 'depth_image_size': self.depth_image_scale}
            data_dict = self.get_image(info, data_dict)

        if self.training:
            # perform augmentations
            data_dict = self.augmentor.forward(data_dict)
            if not data_dict['valid']:
                return ScanNetDataset.__getitem__(self, np.random.randint(self.__len__()))
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

    def get_image(self, info, data_dict, resize_debug=False):
        data_dict['point_img_1d'] = {}
        data_dict['point_img'] = {}
        data_dict['point_img_idx'] = {}
        data_dict['image_shape'] = {}
        scene_name = data_dict['scene_name']
        depth_image_size = info['depth_image_size']
        pose_paths = sorted(
            glob.glob(str(self.root_path / self.image_path / scene_name / 'pose/*.txt')),
            key=lambda a: os.path.basename(a).split('.')
        )
        depth_paths = sorted(
            glob.glob(str(self.root_path / self.image_path / scene_name / 'depth/*.png')),
            key=lambda a: os.path.basename(a).split('.')
        )
        color_paths = sorted(
            glob.glob(str(self.root_path / self.image_path / scene_name / 'color/*.jpg')),
            key=lambda a: os.path.basename(a).split('.')
        )
        assert len(pose_paths) == len(depth_paths) and len(pose_paths) == len(color_paths)
        try:
            depth_intrinsic = np.loadtxt(
                os.path.join(self.root_path, self.image_path, scene_name, 'intrinsics_depth.txt')
            )
        except:
            depth_intrinsic = np.loadtxt(
                os.path.join(self.root_path, self.image_path, scene_name, 'intrinsic_depth.txt')
            )
        points_xyz = data_dict['points_xyz'] + np.array(self.pc_means[scene_name]).reshape(-1, 3)
        if resize_debug:
            target_size = (120, 160)
            scale = (480 - 1) / (target_size[0] - 1)
            depth_intrinsic[:1, :] = depth_intrinsic[:1, :] / scale
            depth_intrinsic[1:2, :] = depth_intrinsic[1:2, :] / scale
            depth_image_size = target_size
        if depth_image_size[0] != 480:
            scale = (480 - 1) / (depth_image_size[0] - 1)
            depth_intrinsic[:1, :] = depth_intrinsic[:1, :] / scale
            depth_intrinsic[1:2, :] = depth_intrinsic[1:2, :] / scale

        for ind, (pose, depth, color) in enumerate(zip(pose_paths, depth_paths, color_paths)):
            image_name = pose.split('/')[-1].split('.')[0]
            point_idx, image_idx_1d, image_idx, color_image_shape = self.project_point_to_image(
                points_xyz, pose, depth, color, depth_image_size, depth_intrinsic)
            data_dict['point_img_1d'][image_name.lower()] = image_idx_1d
            data_dict['point_img'][image_name.lower()] = image_idx
            data_dict['point_img_idx'][image_name.lower()] = point_idx
            data_dict['image_shape'][image_name.lower()] = color_image_shape

        data_dict['depth_image_size'] = depth_image_size
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

    def __del__(self):
        if not self.cache:
            return

        for item in self.data_list:
            if os.path.exists("/dev/shm/scannet_{}".format(item.split('/')[-1][:-4] + '_xyz')):
                sa_delete("shm://scannet_{}".format(item.split('/')[-1][:-4] + '_rgb'))
                sa_delete("shm://scannet_{}".format(item.split('/')[-1][:-4] + '_xyz'))
                if self.split_file.find('test') < 0:
                    sa_delete("shm://{}".format(item.split('/')[-1][:-4] + '_label'))


class ScanNetInstDataset(ScanNetDataset):
    def __init__(self, dataset_cfg, class_names, training, root_path, logger=None):
        ScanNetDataset.__init__(self, dataset_cfg, class_names, training, root_path, logger=logger)
        self.inst_class_idx = dataset_cfg.inst_class_idx
        self.inst_label_shift = dataset_cfg.inst_label_shift
        if 'base_inst_class_idx' in dataset_cfg:
            self.base_inst_class_idx = dataset_cfg.base_inst_class_idx
            self.novel_inst_class_idx = dataset_cfg.novel_inst_class_idx
        elif 'base_class_idx' in dataset_cfg:
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
            return ScanNetInstDataset.__getitem__(self, np.random.randint(self.__len__()))
        info = self.get_inst_info(points, inst_label.astype(np.int32), label)
        data_dict['inst_label'] = inst_label
        data_dict.update(info)
        return data_dict

    def get_inst_info(self, xyz, instance_label, semantic_label):
        ret = super().get_inst_info(xyz, instance_label, semantic_label)
        ret['inst_cls'] = [x - self.inst_label_shift if x != -100 else x for x in ret['inst_cls']]
        return ret

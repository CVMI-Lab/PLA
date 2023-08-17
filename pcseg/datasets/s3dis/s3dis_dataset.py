import SharedArray as SA
import os
import numpy as np
import pickle
import json
import torch
import glob
import cv2

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
                return S3DISDataset.__getitem__(self, np.random.randint(self.__len__()))
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

    def get_image(self, info, data_dict):
        data_dict['point_img_1d'] = {}
        data_dict['point_img'] = {}
        data_dict['point_img_idx'] = {}
        data_dict['image_shape'] = {}
        scene_name = data_dict['scene_name']
        depth_image_size = info['depth_image_size']

        _, area_num, room_name, room_num = scene_name.split('_')
        pose_paths = sorted(
            glob.glob(str(self.root_path / self.image_path / f'area_{area_num}' /
                          'data/pose/*_{}_{}_*.json'.format(room_name, room_num))),
            key=lambda a: os.path.basename(a).split('.')
        )
        depth_paths = sorted(
            glob.glob(str(self.root_path / self.image_path / f'area_{area_num}' /
                          'data/depth/*_{}_{}_*.png'.format(room_name, room_num))),
            key=lambda a: os.path.basename(a).split('.')
        )
        color_paths = sorted(
            glob.glob(str(self.root_path / self.image_path / f'area_{area_num}' /
                          'data/rgb/*_{}_{}_*.png'.format(room_name, room_num))),
            key=lambda a: os.path.basename(a).split('.')
        )
        try:
            assert len(pose_paths) == len(depth_paths) and len(pose_paths) == len(color_paths)
        except:  # ignore this sample
            pose_paths = depth_paths = color_paths = []

        points_xyz = data_dict['points_xyz'] + np.array(self.pc_mins[scene_name]).reshape(-1, 3)

        for ind, (pose, depth, color) in enumerate(zip(pose_paths, depth_paths, color_paths)):
            image_name = pose.split('/')[-1].split('.')[0][:-len('_domain_pose')]
            # print(image_name)
            point_idx, image_idx_1d, image_idx, color_image_shape = \
                self.project_point_to_image(points_xyz, pose, depth, color, depth_image_size)
            data_dict['point_img_1d'][image_name.lower()] = image_idx_1d
            data_dict['point_img'][image_name.lower()] = image_idx
            data_dict['point_img_idx'][image_name.lower()] = point_idx
            data_dict['image_shape'][image_name.lower()] = color_image_shape

        data_dict['depth_image_size'] = depth_image_size
        return data_dict

    @staticmethod
    def project_point_to_image(points_world, pose_path, depth_path, color_path, image_size):

        with open(pose_path, 'r') as fin:
            data = json.load(fin)
        depth_intrinsic = np.array(data['camera_k_matrix'])

        fx = depth_intrinsic[0, 0]
        fy = depth_intrinsic[1, 1]
        cx = depth_intrinsic[0, 2]
        cy = depth_intrinsic[1, 2]
        bx = 0
        by = 0

        # == processing depth ===
        depth_img = cv2.imread(depth_path, -1)  # read 16bit grayscale image
        depth_shift = 512.0
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
        pose = data['camera_rt_matrix']
        pose.append([0, 0, 0, 1])
        pose = np.array(pose)
        # pose = np.linalg.inv(pose)[:3]

        # == 3D to camera coordination ===
        points = np.hstack((points_world[..., :3], np.ones((points_world.shape[0], 1))))
        # points = np.dot(points, np.linalg.inv(np.transpose(pose)))
        points = np.dot(points, np.transpose(pose))

        # == camera to image coordination ===
        u = (points[..., 0] - bx) * fx / points[..., 2] + cx
        v = (points[..., 1] - by) * fy / points[..., 2] + cy
        d = points[..., 2]
        u = (u + 0.5).astype(np.int32)
        v = (v + 0.5).astype(np.int32)

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

    def _del__(self):
        if not self.cache:
            return

        for item in self.data_list:
            if os.path.exists("/dev/shm/{}".format(item)):
                sa_delete("shm://{}".format(item))


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
            return S3DISInstDataset.__getitem__(self, np.random.randint(self.__len__()))
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

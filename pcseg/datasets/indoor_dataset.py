import os
import pickle
import copy
import numpy as np

from .dataset import DatasetTemplate
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor


class IndoorDataset(DatasetTemplate):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super(IndoorDataset, self).__init__(dataset_cfg, class_names, training, root_path, logger=logger)

        self.repeat = dataset_cfg.DATA_PROCESSOR.repeat
        self.voxel_scale = dataset_cfg.DATA_PROCESSOR.voxel_scale
        self.max_npoint = dataset_cfg.DATA_PROCESSOR.max_npoint
        self.full_scale = dataset_cfg.DATA_PROCESSOR.full_scale
        self.point_range = dataset_cfg.DATA_PROCESSOR.point_range
        self.voxel_mode = dataset_cfg.DATA_PROCESSOR.voxel_mode
        self.rgb_norm = dataset_cfg.DATA_PROCESSOR.rgb_norm
        self.cache = dataset_cfg.DATA_PROCESSOR.cache
        self.downsampling_scale = dataset_cfg.DATA_PROCESSOR.get('downsampling_scale', 1)

        self.augmentor = DataAugmentor(
            self.dataset_cfg,
            **{
                'ignore_label': self.ignore_label,
                'voxel_scale': self.voxel_scale,
                'full_scale': self.full_scale,
                'max_npoint': self.max_npoint,
            }
        )

        self.voxel_size = [1.0 / self.voxel_scale, 1.0 / self.voxel_scale, 1.0 / self.voxel_scale]

        num_point_features = 0
        if dataset_cfg.DATA_PROCESSOR.xyz_as_feat:
            num_point_features += 3

        if dataset_cfg.DATA_PROCESSOR.rgb_as_feat:
            num_point_features += 3

        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range, training=self.training,
            num_point_features=num_point_features
        )

    @staticmethod
    def visualize_o3d(xyz, label=None, color=None, density=None, dataset='scannet', origin=False):
        import open3d
        import random
        from operator import itemgetter
        if dataset == 'scannet':
            from tools.visual_utils.open3d_vis_utils import SCANNET_CLASS_COLOR as CLASS_COLOR, \
                SCANNET_DA_SEMANTIC_NAMES as SEMANTIC_NAMES
        elif dataset == 's3dis':
            from tools.visual_utils.open3d_vis_utils import S3DIS_CLASS_COLOR as CLASS_COLOR, \
                S3DIS_DA_SEMANTIC_NAMES as SEMANTIC_NAMES
        else:
            raise NotImplementedError
        pcd = open3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(xyz[(label >= 0) & (label != 255)])
        if label is not None:
            try:
                label_color = np.array(
                    itemgetter(*SEMANTIC_NAMES[label[(label >= 0) & (label != 255)].astype(np.int64)])
                    (CLASS_COLOR))
                pcd.points = open3d.utility.Vector3dVector(xyz[(label >= 0) & (label != 255)])
                pcd.colors = open3d.utility.Vector3dVector(label_color / 255.0)
            except IndexError:  # assign random colors
                num_labels = set(label)
                label_color = np.zeros_like(xyz)
                random_color = lambda: random.randint(0, 255)
                for i_com in num_labels:
                    label_color[label == i_com, :] = [random_color(), random_color(), random_color()]
                pcd.points = open3d.utility.Vector3dVector(xyz)
                pcd.colors = open3d.utility.Vector3dVector(label_color / 255.0)
        if color is not None:
            pcd.points = open3d.utility.Vector3dVector(xyz)
            pcd.colors = open3d.utility.Vector3dVector(color / 255.0)
        if density is not None:
            jet_map = np.loadtxt('jet_int.txt', dtype=np.int)
            color = jet_map[np.round(density * 255).astype(np.int)]
            pcd.points = open3d.utility.Vector3dVector(xyz)
            pcd.colors = open3d.utility.Vector3dVector(color / 255.0)
        if label is None and color is None and density is None:
            pcd.points = open3d.utility.Vector3dVector(xyz)
        if origin:
            original_point = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
            open3d.visualization.draw_geometries([original_point, pcd])
        else:
            open3d.visualization.draw_geometries([pcd])

    @staticmethod
    def filter_by_index(e_list, idx):
        filtered_e_list = list()
        for e in e_list:
            filtered_e_list.append(e[idx])
        return filtered_e_list

    @staticmethod
    def subsample(xyz, label, ds_scale):
        # subsample. Notice that per-class subsampling will automatically ignore ignore_label
        if isinstance(ds_scale, list):
            subsample_idx = np.zeros(0, dtype=np.int64)
            for i, ds in enumerate(ds_scale):
                _idx = np.where(label == i)[0]
                _subsample_idx = np.random.choice(_idx, len(_idx), replace=False)[:int(len(_idx) / ds_scale[i])]
                subsample_idx = np.concatenate((subsample_idx, _subsample_idx))
        else:
            subsample_idx = np.random.choice(xyz.shape[0], xyz.shape[0], replace=False)[:int(xyz.shape[0] / ds_scale)]
        subsample_idx.sort()
        return subsample_idx

    # instance seg
    def get_valid_inst_label(self, inst_label, valid_mask):
        inst_label[~valid_mask] = self.ignore_label
        label_set = np.unique(inst_label[inst_label >= 0])
        remapper = np.full((1000,), self.ignore_label)
        remapper[label_set.astype(np.int64)] = np.arange(len(label_set))
        inst_label[inst_label >= 0] = remapper[inst_label[inst_label >= 0].astype(np.int64)]
        # inst_label[~valid_mask] = self.ignore_label
        # j = 0
        # while (j < inst_label.max()):
        #     if (len(np.where(inst_label == j)[0]) == 0):
        #         inst_label[inst_label == inst_label.max()] = j
        #     j += 1
        return inst_label 

    def get_inst_info(self, xyz, inst_label, semantic_label):

        inst_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * self.ignore_label 
        # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        inst_pointnum = []   # (nInst), int
        inst_cls = []
        inst_num = int(inst_label.max()) + 1
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
        return {'inst_num': inst_num, 'inst_info': inst_info, 'inst_pointnum': inst_pointnum, \
                'inst_cls': inst_cls, 'pt_offset_label': pt_offset_label}

    # caption func
    def include_caption_infos(self):
        """
        scene_image_corr_dict = {
            scene_name : {
                image_name: np.array [pts_index, ...] for the given image in the scene
            }
        }
        """
        if self.caption_cfg.get('VIEW', None) and self.caption_cfg.VIEW.ENABLED:
            scene_image_corr_infos = pickle.load(
                open(self.root_path / self.caption_cfg.VIEW.IMAGE_CORR_PATH, 'rb')
            )
        else:
            scene_image_corr_infos = None

        if self.caption_cfg.get('ENTITY', None) and self.caption_cfg.ENTITY.ENABLED:
            scene_image_corr_entity_infos = pickle.load(
                open(self.root_path / self.caption_cfg.ENTITY.IMAGE_CORR_PATH, 'rb')
            )
        else:
            scene_image_corr_entity_infos = None

        return scene_image_corr_infos, scene_image_corr_entity_infos

    def get_caption_image_corr_and_name_from_memory(self, scene_name, index):
        image_name_dict = {}
        image_corr_dict = {}

        if self.caption_cfg.get('SCENE', None) and self.caption_cfg.SCENE.ENABLED:
            image_name_dict['scene'] = None
            image_corr_dict['scene'] = None

        if hasattr(self, 'scene_image_corr_infos') and self.scene_image_corr_infos is not None:
            if isinstance(self.scene_image_corr_infos, dict):
                # assert scene_name in self.scene_image_corr_infos
                info = self.scene_image_corr_infos.get(scene_name, {})
            else:
                cur_caption_idx = copy.deepcopy(self.scene_image_corr_infos[index])
                assert scene_name == cur_caption_idx['scene_name']
                info = cur_caption_idx['infos']
            if len(info) > 0:
                image_name_view, image_corr_view = zip(*info.items())
            else:
                image_name_view, image_corr_view = [], []
            image_name_dict['view'] = image_name_view
            image_corr_dict['view'] = image_corr_view

        if hasattr(self, 'scene_image_corr_entity_infos') and self.scene_image_corr_entity_infos is not None:
            if isinstance(self.scene_image_corr_entity_infos, dict):
                # assert scene_name in self.scene_image_corr_entity_infos
                info = self.scene_image_corr_entity_infos.get(scene_name, {})
            else:
                cur_caption_idx = copy.deepcopy(self.scene_image_corr_entity_infos[index])
                assert scene_name == cur_caption_idx['scene_name']
                info = cur_caption_idx['infos']
            if len(info) > 0:
                image_name_entity, image_corr_entity = zip(*info.items())
            else:
                image_name_entity, image_corr_entity = [], []
            image_name_dict['entity'] = image_name_entity
            image_corr_dict['entity'] = image_corr_entity

        return image_corr_dict, image_name_dict

    def get_caption_image_corr_and_name_from_file(self, scene_name):
        image_name_dict = {}
        image_corr_dict = {}

        if self.caption_cfg.get('SCENE', None) and self.caption_cfg.SCENE.ENABLED:
            image_name_dict['scene'] = None
            image_corr_dict['scene'] = None

        if self.caption_cfg.get('VIEW', None) and self.caption_cfg.VIEW.ENABLED:
            path = self.root_path / self.caption_cfg.VIEW.IMAGE_CORR_PATH / (scene_name + '.pickle')
            if os.path.exists(path):
                info = pickle.load(open(path, 'rb'))
            else:
                info = {}
            if len(info) > 0:
                image_name_view, image_corr_view = zip(*info.items())
            else:
                image_name_view = image_corr_view = []
            image_name_dict['view'] = image_name_view
            image_corr_dict['view'] = image_corr_view

        if self.caption_cfg.get('ENTITY', None) and self.caption_cfg.ENTITY.ENABLED:
            path = self.root_path / self.caption_cfg.ENTITY.IMAGE_CORR_PATH / (scene_name + '.pickle')
            if os.path.exists(path):
                info = pickle.load(open(path, 'rb'))
            else:
                info = {}
            if len(info) > 0:
                image_name_entity, image_corr_entity = zip(*info.items())
            else:
                image_name_entity = image_corr_entity = []
            image_name_dict['entity'] = image_name_entity
            image_corr_dict['entity'] = image_corr_entity

        return image_corr_dict, image_name_dict

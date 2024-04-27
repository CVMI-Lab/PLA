import os
import copy
import pickle
import numpy as np
import cv2

from pathlib import Path

import tqdm
from skimage import io

from ..outdoor_dataset import OutdoorDataset
from ...utils import common_utils, caption_utils
from . import label_mapping, nuscenes_utils


class NuScenesDataset(OutdoorDataset):
    def __init__(self,  dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None, split=None):
        root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger, split=split
        )
        self.infos = []
        self.include_nuscenes_data(self.mode)
        self.filename_list = [ii['lidar_path'].split('/')[-1] for ii in self.infos]
        self.point_caption_idx, self.entity_point_caption_idx = self.include_point_caption_idx()

        # for image loading
        self.with_data_inst = self.dataset_cfg.get('WITH_DATA_INST', None)
        self.load_image = self.dataset_cfg.get('LOAD_IMAGE', None)
        self.image_list = self.dataset_cfg.get('IMAGE_LIST', None)
        self.image_scale = self.dataset_cfg.get('IMAGE_SCALE', None)
        if self.with_data_inst:
            from nuscenes.nuscenes import NuScenes
            self.data_inst = NuScenes(version=dataset_cfg.VERSION, dataroot=root_path, verbose=True)

    def include_nuscenes_data(self, mode):
        self.logger.info('Loading NuScenes dataset')
        nuscenes_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            if self.oss_client:
                info_path = os.path.join(self.oss_root_path, info_path)
            else:
                info_path = self.root_path / info_path

            if not common_utils.check_exists(info_path):
                continue

            nuscenes_infos = pickle.load(self.oss_client.get(info_path)) if self.oss_client else pickle.load(open(info_path, 'rb'))

        self.infos.extend(nuscenes_infos)
        self.logger.info('Total samples for NuScenes dataset: %d' % (len(nuscenes_infos)))

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def get_data_list(self):
        return self.filename_list

    def get_lidar(self, info):
        if self.oss_client:
            lidar_path = os.path.join(self.oss_root_path, info['lidar_path'])
            points = np.frombuffer(self.oss_client.get(lidar_path), dtype=np.float32, count=-1).reshape(-1, 5)[:, :4]
        else:
            lidar_path = self.root_path / info['lidar_path']
            points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        return points

    def get_image(self, info, data_dict):
        """Modified from det3d/datasets/pipelines/loading.py

        Args:
            info (dict): data info
            data_dict (dict): 
        """
        assert self.with_data_inst
        sample_token = info['token']
        sample_record = self.data_inst.get('sample', sample_token)
        data_dict['calib'], data_dict['image_shape'], data_dict['cam'] = {}, {}, {}
        data_dict['image_path'] = {}
        if self.dataset_cfg.get('LOAD_POINT_TO_IMAGE_IDX', None):
            data_dict['point_img'], data_dict['point_img_idx'] = {}, {}
        for cam_channel in self.image_list:
            camera_token = sample_record['data'][cam_channel]
            cam = self.data_inst.get('sample_data', camera_token)

            image_path = os.path.join(self.data_inst.dataroot, cam['filename'])
            data_dict['image_path'][cam_channel.lower()] = image_path
            if self.dataset_cfg.get('READ_IMAGE', False):
                cam_img = self.read_image(image_path)
                if self.image_scale != 1:
                    new_shape = [int(cam_img.shape[1]*self.image_scale), int(cam_img.shape[0]*self.image_scale)]
                    cam_img = cv2.resize(cam_img, new_shape)
                data_dict['image'][cam_channel.lower()] = cam_img
                cam_img_shape = cam_img.shape
            else:
                cam_img = None
                cam_img_shape = tuple(self.dataset_cfg.DEFAULT_IMAGE_SHAPE)

            data_dict['image_scale'] = self.image_scale
            data_dict['image_shape'][cam_channel.lower()] = cam_img_shape
            pointsensor_token = sample_record['data']['LIDAR_TOP']
            lidar2cam, cam_intrinsic = nuscenes_utils.get_lidar2cam_matrix(self.data_inst, pointsensor_token, cam)
            cam_key = 'lidar2cam'+cam_channel.lstrip('CAM').lower()
            intri_key = 'cam_intrinsic'+cam_channel.lstrip('CAM').lower()
            data_dict['calib'][cam_key] = lidar2cam
            data_dict['calib'][intri_key] = cam_intrinsic

            if self.dataset_cfg.get('LOAD_POINT_TO_IMAGE_IDX', None):
                point_img, point_img_idx = self.project_point_to_image(
                    cam_channel, cam_img, cam_img_shape, data_dict['points'], lidar2cam, cam_intrinsic
                )
                data_dict['point_img'][cam_channel.lower()] = point_img
                data_dict['point_img_idx'][cam_channel.lower()] = point_img_idx

        return data_dict

    def project_point_to_image(self, cam_key, image, image_shape, points, lidar2cam, cam_intrinsic):
        points_lidar = points.copy()
        assert points_lidar.shape[-1] == 4

        points_lidar[:, -1] = 1
        # lidar to camera projection
        point_camera = np.dot(lidar2cam, points_lidar.transpose())
        depth = point_camera[2, :].copy()

        point_img = self.view_points(point_camera[:3, :], np.array(cam_intrinsic), normalize=True)
        if self.image_scale != 1:
            point_img = (self.image_scale * point_img.astype(np.float32)).astype(np.int)

        # filter invalid points for currect image
        mask = (point_img[0] > 1) & (point_img[1] > 1) & (point_img[0] < image_shape[1]-1) & (point_img[1] < image_shape[0]-1) & (depth > 1)
        point_img = point_img.transpose()[mask, :2].astype(np.int)
        point_img_idx = mask.nonzero()[0]

        # debug visualization
        # image_test = (image * 255).astype(np.uint8())
        # image_test = np.ascontiguousarray(image_test)
        #
        # for _point in point_img:
        #     if _point.sum() > 0:
        #         circle_coord = tuple(_point)
        #         cv2.circle(image_test, circle_coord, 3, (0, 255, 0), -1)

        # import ipdb; ipdb.set_trace(context=20)
        # cv2.imwrite(f'../debug_image/{cam_key}.png', image_test)

        return point_img, point_img_idx

    @staticmethod
    def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
        # copied from https://github.com/nutonomy/nuscenes-devkit/
        # only for debug use
        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[0] == 3

        viewpad = np.eye(4)
        viewpad[:view.shape[0], :view.shape[1]] = view

        nbr_points = points.shape[1]

        # Do operation in homogenous coordinates.
        points = np.concatenate((points, np.ones((1, nbr_points))))
        points = np.dot(viewpad, points)
        points = points[:3, :]

        if normalize:
            points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

        return points

    def read_image(self, path):
        if self.oss_client:
            image = io.imread(self.oss_client.get(path))
        else:
            image = io.imread(path)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_label(self, info):
        if self.oss_client:
            label_path = os.path.join(self.oss_root_path, info['lidarseg_label_path'])
            origin_labels = np.frombuffer(self.oss_client.get(label_path), dtype=np.uint8, count=-1).reshape([-1])
        else:
            label_path = self.root_path / info['lidarseg_label_path']
            origin_labels = np.fromfile(str(label_path), dtype=np.uint8, count=-1).reshape([-1])

        mapped_labels = np.vectorize(label_mapping.learning_map.get)(origin_labels)
        mapped_labels[mapped_labels == 0] = self.ignore_label + 1
        mapped_labels = mapped_labels - 1
        labels = mapped_labels.astype(np.uint8)

        return labels

    def map_label_to_ov_labels(self, labels):
        if hasattr(self, 'base_class_mapper'):
            binary_labels = self.binary_class_mapper[labels.astype(np.int64)].astype(np.float32)
        else:
            binary_labels = np.ones_like(labels)

        # base / novel label
        if self.class_mode == 'base':
            labels = self.base_class_mapper[labels.astype(np.int64)]
        elif self.class_mode == 'all' and hasattr(self, 'ignore_class_idx'):
            labels = self.valid_class_mapper[labels.astype(np.int64)]

        return labels, binary_labels

    def get_caption_image_corr_and_name(self, index, scene_name):
        if self.point_caption_idx is not None or self.entity_point_caption_idx is not None:
            return self.get_caption_image_corr_and_name_from_info(index, scene_name)
        else:
            return self.get_caption_image_corr_and_name_separately(scene_name)

    def get_caption_image_corr_and_name_from_info(self, index, scene_name):
        image_name_dict = {}
        image_corr_dict = {}

        if self.need_scene_caption:
            image_name_dict['scene'] = None
            image_corr_dict['scene'] = None

        if self.need_view_caption:
            # use deep copy to avoid open too many files
            cur_caption_idx = copy.deepcopy(self.point_caption_idx[index])
            assert scene_name == cur_caption_idx['scene_name']
            if len(cur_caption_idx['infos']) > 0:
                image_name_view, image_corr_view = zip(*cur_caption_idx['infos'].items())
            else:
                image_name_view = image_corr_view = []
            image_name_dict['view'] = image_name_view
            image_corr_dict['view'] = image_corr_view

        if self.need_entity_caption:
            cur_entity_point_caption_idx = copy.deepcopy(self.entity_point_caption_idx[index])
            assert scene_name == cur_entity_point_caption_idx['scene_name']
            if len(cur_entity_point_caption_idx['infos']) > 0:
                image_name_entity, image_corr_entity = zip(*cur_entity_point_caption_idx['infos'].items())
            else:
                image_name_entity = image_corr_entity = []
            image_name_dict['entity'] = image_name_entity
            image_corr_dict['entity'] = image_corr_entity

        return image_corr_dict, image_name_dict

    def get_caption_image_corr_and_name_separately(self, scene_name):
        image_name_dict = {}
        image_corr_dict = {}

        if self.need_scene_caption:
            image_name_dict['scene'] = None
            image_corr_dict['scene'] = None

        if self.need_view_caption:
            if self.oss_client:
                scene_caption_idx_path = os.path.join(self.oss_root_path, 'caption_idx', scene_name + '.pkl')
                info = pickle.load(self.oss_client.get(scene_caption_idx_path))
            else:
                scene_caption_idx_path = self.root_path / 'caption_idx' / (scene_name + '.pkl')
                info = pickle.load(open(scene_caption_idx_path, 'rb'))
            if len(info) > 0:
                image_name_view, image_corr_view = zip(*info.items())
            else:
                image_name_view = image_corr_view = []
            image_name_dict['view'] = image_name_view
            image_corr_dict['view'] = image_corr_view

        if self.need_entity_caption:
            if self.oss_client:
                entity_caption_idx_path = os.path.join(self.root_path, 'caption_entity_idx', scene_name + '.pkl')
                info = pickle.load(self.oss_client.get(entity_caption_idx_path))
            else:
                entity_caption_idx_path = self.root_path / 'caption_entity_idx' / (scene_name + '.pkl')
                info = pickle.load(open(entity_caption_idx_path, 'rb'))
            if len(info) > 0:
                image_name_entity, image_corr_entity = zip(*info.items())
            else:
                image_name_entity = image_corr_entity = []
            image_name_dict['entity'] = image_name_entity
            image_corr_dict['entity'] = image_corr_entity

        return image_corr_dict, image_name_dict

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar(info)
        labels = self.get_label(info)

        labels, binary_labels = self.map_label_to_ov_labels(labels)

        # get captioning data
        if self.training or self.dataset_cfg.get('FILTER_WITH_N_CAPTIONS', -1) != -1:
            scene_name = info['lidar_path'].split('.')[0].replace('/', '_')
            image_corr_dict, image_name_dict = self.get_caption_image_corr_and_name(index, scene_name)
            caption_data = self.select_caption_and_idx_all(scene_name, image_name_dict, image_corr_dict)
        else:
            caption_data = None
        
        data_dict = {
            'points': points.copy(),
            'labels': labels.copy(),
            'binary_labels': binary_labels,
            'scene_name': info['lidar_path'].split('.')[0].replace('/', '_'),
            'metadata': {'token': info['token']},
            'ids': index,
            'caption_data': caption_data,
            'pc_count': points.shape[0],
            'origin_idx': np.arange(points.shape[0]).astype(np.int64)
        }

        # filtering points
        if self.dataset_cfg.get('FILTER_WITH_N_CAPTIONS', -1) != -1:
            n_captions_points = caption_utils.n_captions_for_points(image_corr_dict, points.shape[0])
            data_dict['pred_mask'] = n_captions_points == self.dataset_cfg.FILTER_WITH_N_CAPTIONS
            if hasattr(self, 'need_n_captions_points'):
                data_dict['n_captions_points'] = n_captions_points

        if self.load_image:
            data_dict = self.get_image(info, data_dict)

        # data augmentation
        if self.training:
            data_dict = self.augmentor.forward(data_dict)

        data_dict['feats'] = data_dict['points']
        data_dict = self.data_processor.forward(data_dict)

        if self.dataset_cfg.get('XYZ_NORM', True):
            data_dict['points'] -= data_dict['points'].min(0)

        # visualization debug code
        # import tools.visual_utils.open3d_vis_utils as vis
        # vis_dict = {
        #     'points': data_dict['points'],
        #     'point_size': 2.0
        # }
        # vis.dump_vis_dict(vis_dict)
        # import ipdb;
        # ipdb.set_trace(context=20)

        return data_dict


class NuScenesPanopticDataset(NuScenesDataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None, split=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger, split)
        self.inst_class_idx = dataset_cfg.inst_class_idx
        self.stuff_class_idx = dataset_cfg.stuff_class_idx
        self.inst_label_shift = dataset_cfg.inst_label_shift
        if 'base_class_idx' in dataset_cfg:
            # panoptic seg, inst first
            base_inst_class_idx = np.array(list(set(self.base_class_idx) & set(self.inst_class_idx)))
            novel_inst_class_idx = np.array(list(set(self.novel_class_idx) & set(self.inst_class_idx)))
            self.base_inst_class_idx = base_inst_class_idx - self.inst_label_shift
            self.novel_inst_class_idx = novel_inst_class_idx - self.inst_label_shift
        self.sem2ins_classes = dataset_cfg.sem2ins_classes
        self.min_gt_pts = dataset_cfg.get('MIN_GT_POINTS', 0)

    def get_panoptic_label(self, info):
        if self.oss_client:
            label_path = os.path.join(self.oss_root_path, info['lidarseg_label_path'].replace('lidarseg', 'panoptic').replace('.bin', '.npz'))
            labels = np.load(self.oss_client.get(label_path))['data']
        else:
            label_path = self.root_path / info['lidarseg_label_path'].replace('lidarseg', 'panoptic').replace('.bin', '.npz')
            labels = nuscenes_utils.load_bin_file(label_path, type='panoptic')
        sem_label = labels // 1000
        mapped_labels = np.vectorize(label_mapping.learning_map.get)(sem_label)
        mapped_labels[mapped_labels == 0] = self.ignore_label + 1
        mapped_labels = mapped_labels - 1
        sem_label = mapped_labels.astype(np.uint8)

        inst_label = labels % 1000
        inst_label[inst_label == 0] = self.ignore_label
        inst_label = inst_label.astype(np.int64)

        return sem_label, inst_label

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        points = self.get_lidar(info)
        # labels = self.get_label(info)
        labels, inst_label = self.get_panoptic_label(info)

        labels, binary_labels = self.map_label_to_ov_labels(labels)
        inst_label[labels == self.ignore_label] = self.ignore_label

        # get captioning data
        if self.training or self.dataset_cfg.get('FILTER_WITH_N_CAPTIONS', -1) != -1:
            scene_name = info['lidar_path'].split('.')[0].replace('/', '_')
            image_corr_dict, image_name_dict = self.get_caption_image_corr_and_name(index, scene_name)
            caption_data = self.select_caption_and_idx_all(scene_name, image_name_dict, image_corr_dict)
        else:
            caption_data = None
        
        data_dict = {
            'points': points.copy(),
            'labels': labels.copy(),
            'inst_label': inst_label.copy(),
            'binary_labels': binary_labels,
            'scene_name': info['lidar_path'].split('.')[0].replace('/', '_'),
            'metadata': {'token': info['token']},
            'ids': index,
            'caption_data': caption_data,
            'pc_count': points.shape[0],
            'origin_idx': np.arange(points.shape[0]).astype(np.int64)
        }

        # === instance pseudo offset label ====
        if self.training and hasattr(self, 'pseudo_label_dir'):
            # print(self.pseudo_label_dir)
            fn = self.filename_list[index]
            pseudo_offset = self.load_pseudo_labels(fn.split('/')[-1].split('.')[0], dtype=np.float, shape=(-1, 3))
            data_dict['pt_offset_mask'] = (pseudo_offset == 0).sum(1) != 3
            # pseudo_offset[(pseudo_offset == 0).sum(1) == 3] = -100.
            data_dict['pseudo_offset_target'] = data_dict['points'][..., :3] + pseudo_offset

        # filtering points
        if self.dataset_cfg.get('FILTER_WITH_N_CAPTIONS', -1) != -1:
            n_captions_points = caption_utils.n_captions_for_points(image_corr_dict, points.shape[0])
            data_dict['pred_mask'] = n_captions_points == self.dataset_cfg.FILTER_WITH_N_CAPTIONS
            if hasattr(self, 'need_n_captions_points'):
                data_dict['n_captions_points'] = n_captions_points

        if self.load_image:
            data_dict = self.get_image(info, data_dict)

        # data augmentation
        if self.training:
            data_dict = self.augmentor.forward(data_dict)

        data_dict['feats'] = data_dict['points']
        data_dict = self.data_processor.forward(data_dict)

        if self.dataset_cfg.get('XYZ_NORM', True):
            data_dict['points'] -= data_dict['points'].min(0)

        # inst info collection
        data_dict['points_xyz'] = data_dict['points'][..., :3]
        points, label, inst_label, binary_label = \
            data_dict['points_xyz'], data_dict['labels'], data_dict['inst_label'], data_dict['binary_labels']
        if self.training:
            inst_label[binary_label == 0] = self.ignore_label
        inst_label = self.get_valid_inst_label(inst_label, label != self.ignore_label)
        inst_label_mask = self.filter_instance_with_min_points(inst_label, self.min_gt_pts)
        inst_label = self.get_valid_inst_label(inst_label, inst_label_mask)
        if self.training and len(np.unique(inst_label)) == 0 and inst_label[0] == self.ignore_label:
            return NuScenesPanopticDataset.__getitem__(self, np.random.randint(self.__len__()))
        info = self.get_inst_info(points, inst_label.astype(np.int32), label)
        if self.training and hasattr(self, 'pseudo_label_dir'):
            # print('update pseudo label')
            info['pt_offset_label'][binary_label == 0] = (data_dict['pseudo_offset_target'] - points)[binary_label == 0]
            data_dict['pt_offset_mask'] = (data_dict['pt_offset_mask'] & (binary_label == 0)) | (inst_label != self.ignore_label)
            del data_dict['pseudo_offset_target']
        data_dict['inst_label'] = inst_label
        data_dict.update(info)

        return data_dict

def create_nuscenes_info(version, data_path, save_path, max_sweeps=1):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from . import nuscenes_utils
    data_path = data_path / version
    save_path = save_path / version

    assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = nuscenes_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = nuscenes_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps
    )

    if version == 'v1.0-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f:
            pickle.dump(val_nusc_infos, f)


def create_nusences_iterator(model_cfg, dataset_cfg, data_path):
    dataset = NuScenesDataset(
        dataset_cfg=dataset_cfg, class_names=model_cfg.CLASS_NAMES,
        root_path=Path(data_path),
        training=True, logger=common_utils.create_logger()
    )

    for idx, info in tqdm.tqdm(enumerate(dataset.infos), total=len(dataset.infos)):
        scene_name = info['lidar_path'].split('.')[0].replace('/', '_')

        points = dataset.get_lidar(info)
        data_dict = {
            'points': points,
            'scene_name': scene_name,
        }

        data_dict = dataset.get_image(info, data_dict)

        yield data_dict


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--model_cfg', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_nuscenes_infos', 
                        choices=['create_nuscenes_infos'],
                        help='')

    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    args = parser.parse_args()

    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    dataset_cfg.VERSION = args.version
    if args.func == 'create_nuscenes_infos':
        create_nuscenes_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'nuscenes',
            save_path=ROOT_DIR / 'data' / 'nuscenes',
            max_sweeps=dataset_cfg.MAX_SWEEPS,
        )
    else:
        raise NotImplementedError

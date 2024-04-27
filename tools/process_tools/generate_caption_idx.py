import os
import json

import numpy as np
import tqdm
import pickle
import torch
import cv2

from functools import partial
import concurrent.futures as futures

from pcseg.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
from pcseg.datasets.scannet.scannet_dataset import ScanNetDataset
from pcseg.datasets.kitti.kitti_dataset import KittiDataset
from pcseg.utils import common_utils, caption_utils


class CaptionIdxProcessor(object):
    def __init__(self, dataset):
        self.dataset = dataset

        if hasattr(self.dataset, 'infos'):
            self.infos = self.dataset.infos
        else:
            self.infos = self.dataset.data_list

    def get_lidar(self, info, idx):
        if hasattr(self.dataset, 'get_lidar'):
            points = self.dataset.get_lidar(info)
            scene_name = info['lidar_path'].split('.')[0].replace('/', '_')
            data_dict = {'points': points}
        elif hasattr(self.dataset, 'load_data'):
            xyz, _, _, _, _ = self.dataset.load_data(idx)
            scene_name = self.dataset.get_data_list()[idx].split('/')[-1].split('.')[0]
            info = {'scene_name': scene_name, 'depth_image_size': self.dataset.depth_image_scale, 'idx': idx}
            data_dict = {'points_xyz': xyz, 'scene_name': scene_name}
        else:
            raise NotImplementedError

        return data_dict, scene_name, info

    def create_caption_idx(self, num_workers=16):
        save_path = self.dataset.root_path / 'caption_idx'
        save_path.mkdir(parents=True, exist_ok=True)

        create_caption_idx_single_scene = partial(
            self.create_caption_idx_single,
            save_path=save_path
        )

        for idx, info in tqdm.tqdm(enumerate(self.infos), total=len(self.infos)):
            create_caption_idx_single_scene((info, idx))

    def create_caption_idx_single(self, info_with_idx, save_path):
        info, idx = info_with_idx

        data_dict, scene_name, info = self.get_lidar(info, idx)
        scene_caption_save_path = save_path / (scene_name + '.pkl')

        data_dict = self.dataset.get_image(info, data_dict)
        scene_caption_idx = data_dict['point_img_idx']

        for key, values in scene_caption_idx.items():
            scene_caption_idx[key] = torch.from_numpy(values).int()

        with open(scene_caption_save_path, 'wb') as f:
            pickle.dump(scene_caption_idx, f)

    def create_caption_idx_with_crop(self, save_path, detic_crop_info_path=None, window_size=None,
                                     overlap_ratio=None, num_workers=16):
        if detic_crop_info_path is not None:
            detic_crop_infos = pickle.load(open(detic_crop_info_path, 'rb'))
        else:
            detic_crop_infos = [None] * len(self.infos)

        if window_size is not None:
            window_size = np.array(window_size)
            strides = (int(window_size[0] * (1 - overlap_ratio)), int(window_size[1] * (1 - overlap_ratio)))
            create_caption_idx_single_scene = partial(
                self.create_caption_idx_with_crop_single,
                window_size=window_size,
                strides=strides
            )
        else:
            create_caption_idx_single_scene = self.create_caption_idx_with_crop_single

        with futures.ThreadPoolExecutor(num_workers) as executor:
            detic_crop_caption_idx = list(
                tqdm.tqdm(executor.map(
                    create_caption_idx_single_scene, self.infos, range(len(self.infos)), detic_crop_infos
                ), total=len(self.infos))
            )

        with open(save_path, 'wb') as f:
            pickle.dump(detic_crop_caption_idx, f)

    def create_caption_idx_with_crop_single(self, info, idx, detic_infos=None, window_size=None, strides=None):
        if isinstance(info, dict):
            scene_name = info['lidar_path'].split('.')[0].replace('/', '_')
        else:
            scene_name = info.split('/')[-1].split('.')[0]

        if detic_infos is not None:
            assert detic_infos['scene_name'] == scene_name

        data_dict, scene_name, info = self.get_lidar(info, idx)

        scene_caption_idx = {'scene_name': scene_name, 'infos': {}, 'boxes': {}}
        data_dict = self.dataset.get_image(info, data_dict)

        # for each image
        image_name_list = list(data_dict['point_img_idx'].keys())
        custom_image_shape = None
        if args.use_custom_image_path:
            image_path = os.path.join(args.custom_image_path, scene_name, 'color', image_name_list[0] + '.jpg')
            assert os.path.exists(image_path)
            image = cv2.imread(image_path)
            custom_image_shape = image.shape

        for image_name in image_name_list:
            image_name = image_name.lower()
            image_shape = data_dict['image_shape'][image_name]
            if custom_image_shape is not None:
                image_shape = custom_image_shape

            if detic_infos is not None:
                if image_name in detic_infos['infos']:
                    image_detic_pred = detic_infos['infos'][image_name]
                    boxes = image_detic_pred['boxes']
                else:
                    continue
            else:
                boxes = caption_utils.get_sliding_windows(image_shape, window_size, strides)

            if boxes.shape[0] == 0:
                continue

            point_img = data_dict['point_img'][image_name]  # (x, y)
            if data_dict.get('depth_image_size', None):
                # rescale point image if depth image size not match real image
                scale = np.array(image_shape[:2]) / np.array(data_dict['depth_image_size'])
                point_img = point_img * scale

            point_img_idx = torch.from_numpy(data_dict['point_img_idx'][image_name]).int()
            # enlarge boxes with a ratio
            if args.enlarge_box_ratio > 1:
                boxes = caption_utils.enlarge_boxes_size(
                    boxes, args.enlarge_box_ratio, args.enlarge_boxes_max_thresh, image_shape
                )

            # filter by boxes size
            if args.filter_by_image_size:
                boxes_size = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                boxes_size_max_mask = boxes_size >= args.min_image_crop_area * args.enlarge_box_ratio
                boxes = boxes[boxes_size_max_mask]

            selected_boxes = []
            valid_boxes_counter = 0
            for i in range(boxes.shape[0]):
                y_min, x_min, y_max, x_max = boxes[i]
                x_mask = np.logical_and(x_min < point_img[:, 0], point_img[:, 0] < x_max)
                y_mask = np.logical_and(y_min < point_img[:, 1], point_img[:, 1] < y_max)
                mask = np.logical_and(x_mask, y_mask)
                crop_point_img_idx = point_img_idx[mask]
                # filter empty captions
                if args.filter_empty_caption and crop_point_img_idx.shape[0] == 0:
                    continue
                crop_name = f'{image_name}_{valid_boxes_counter}'
                valid_boxes_counter += 1
                scene_caption_idx['infos'][crop_name] = crop_point_img_idx
                selected_boxes.append(boxes[i])

            scene_caption_idx['boxes'][image_name] = np.array(selected_boxes)

        return scene_caption_idx


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset', type=str, default='nuscenes', help='specify the dataset')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--model_cfg', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_view_caption_idx', help='')

    parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
    parser.add_argument('--workers', type=int, default=8, help='')

    parser.add_argument('--detic_info_path', type=str, help='')
    parser.add_argument('--save_path', type=str, help='')

    # filters
    parser.add_argument('--filter_by_image_size', action='store_true', default=False, help='')
    parser.add_argument('--filter_empty_caption', action='store_true', default=False, help='')

    # for basic crop caption
    parser.add_argument('--window_size', default=(400, 500), type=tuple, help='window size for cropping sub images')
    parser.add_argument('--overlap_ratio', default=0.3, type=float, help='overlap ratio when crop images')
    parser.add_argument('--use_custom_image_shape', action='store_true', default=False, help='')
    parser.add_argument('--custom_image_shape', default=(968, 1296, 3), type=tuple, help='given image shape')
    parser.add_argument('--use_custom_image_path', action='store_true', default=False, help='')
    parser.add_argument('--custom_image_path', default='data/scannetv2/scannet_images_125k_1296', type=str, help='')

    # for detic crop caption
    parser.add_argument('--min_image_crop_area', default=3000, type=int, help='minimal image crop size')
    parser.add_argument('--enlarge_boxes_max_thresh', default=8000, type=int, help='maximum size that dont need a enlarge')
    parser.add_argument('--enlarge_box_ratio', default=1.0, type=float, help='enlarge the box with a ratio')

    parser.add_argument('--oss_data', action='store_true', default=False, help='')

    global args
    args = parser.parse_args()

    ROOT_DIR = (Path(__file__).resolve().parent / '../../').resolve()
    print(ROOT_DIR)
    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    if args.dataset in ['nuscenes']:
        dataset_cfg.VERSION = args.version

    # to load class names
    cfg = EasyDict(yaml.safe_load(open(args.model_cfg)))

    # use oss for data loading
    if args.oss_data or (cfg.get('OSS', None) and cfg.OSS.DATA):
        common_utils.oss_data_client = common_utils.OSSClient()
        print(f'Ceph client initialization with root path at {cfg.DATA_CONFIG.OSS_PATH}')

    if args.dataset == 'nuscenes':
        dataset = NuScenesDataset(
            dataset_cfg=dataset_cfg, class_names=cfg.CLASS_NAMES,
            root_path=ROOT_DIR / 'data' / 'nuscenes',
            training=True, logger=common_utils.create_logger()
        )
    elif args.dataset == 'scannet':
        dataset = ScanNetDataset(
            dataset_cfg=dataset_cfg, class_names=cfg.CLASS_NAMES,
            root_path=ROOT_DIR / 'data' / 'scannetv2',
            training=True, logger=common_utils.create_logger()
        )
    elif args.dataset == 'kitti':
        dataset = KittiDataset(
            dataset_cfg=dataset_cfg, class_names=cfg.CLASS_NAMES,
            root_path=ROOT_DIR / 'data' / 'kitti',
            training=True, logger=common_utils.create_logger()
        )

    processor = CaptionIdxProcessor(dataset)
    if args.func == 'create_view_caption_idx':
        """
        python -m tools.process_tools.generate_caption_idx --dataset nuscenes --func create_view_caption_idx \
        --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset_multimodal.yaml \
        --model_cfg tools/cfgs/nuscenes_models/sparseunet_debug_image.yaml \
        --version v1.0-mini
        """
        processor.create_caption_idx(args.workers)
    elif args.func == 'create_caption_idx_detic_crop':
        """
        python -m tools.process_tools.generate_caption_idx --dataset nuscenes --func create_caption_idx_detic_crop \
        --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset_multimodal.yaml \
        --model_cfg tools/cfgs/nuscenes_models/sparseunet_debug_image.yaml --version v1.0-mini --workers 4 \
        --detic_info_path ../Detic/nuscenes_v1.0-mini_detic_pred_results.pkl \
        --save_path data/nuscenes/v1.0-mini/nuscenes_caption_idx_detic_crop.pkl
        """
        processor.create_caption_idx_with_crop(
            args.save_path, args.detic_info_path, num_workers=args.workers
        )
    elif args.func == 'create_caption_idx_basic_crop':
        """ nuScenes
        python -m pcseg.datasets.nuscenes.nuscenes_caption --dataset nuscenes --func create_caption_idx_basic_crop \
        --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset_multimodal.yaml \
        --model_cfg tools/cfgs/nuscenes_models/sparseunet_debug_image.yaml --version v1.0-mini \
        --workers 16 --save_path data/nuscenes/v1.0-mini/nuscenes_caption_idx_basic_crop.pkl
        """
        """ ScanNet
        python -m tools.process_tools.generate_caption_idx --func create_caption_idx_basic_crop \
        --dataset scannet    --cfg_file tools/cfgs/dataset_configs/scannet_dataset_multimodal.yaml \
        --model_cfg tools/cfgs/scannet_models/spconv_clip_image.yaml  --workers 16 \
        --save_path data/scannetv2/scannet_caption_idx_basic_crop.pkl
        """
        processor.create_caption_idx_with_crop(
            args.save_path, window_size=args.window_size, overlap_ratio=args.overlap_ratio, num_workers=args.workers
        )
    elif args.func == 'create_caption_idx_detic_crop_caption':
        """
        python -m tools.process_tools.generate_caption_idx --dataset nuscenes \
        --func create_caption_idx_detic_crop_caption \
        --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset_multimodal.yaml \
        --model_cfg tools/cfgs/nuscenes_models/sparseunet_debug_image.yaml --version v1.0-mini --workers 16 \
        --detic_info_path ../Detic/nuscenes_v1.0-mini_detic_pred_results.pkl  \
        --save_path data/nuscenes/v1.0-mini/nuscenes_caption_idx_detic_crop_cap.pkl \
        --filter_by_image_size --min_image_crop_area 3600
        ############################
        #  with enlarge box crop ###
        ############################
        python -m pcseg.datasets.nuscenes.nuscenes_caption --func create_nuscenes_caption_idx_detic_crop_caption \
        --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset_multimodal.yaml \
        --model_cfg tools/cfgs/nuscenes_models/sparseunet_debug_image.yaml --version v1.0-mini --workers 16 \
        --detic_info_path ../Detic/nuscenes_v1.0-mini_detic_pred_results.pkl  \
        --save_path data/nuscenes/v1.0-mini/nuscenes_caption_idx_detic_crop_cap_enlarge2.5.pkl \
        --filter_by_image_size --min_image_crop_area 3000 --enlarge_box_ratio 2.5
        """
        processor.create_caption_idx_with_crop(
            args.save_path, args.detic_info_path, num_workers=args.workers
        )
    else:
        raise NotImplementedError

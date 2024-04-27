import numpy as np
import tqdm
import pickle
import torch

from functools import partial
import concurrent.futures as futures

from .nuscenes_dataset import NuScenesDataset
from pcseg.utils import common_utils, caption_utils


class NuScenesDatasetCaption(NuScenesDataset):
    def create_caption_idx(self, num_workers=16):
        save_path = self.root_path / 'caption_idx'
        save_path.mkdir(parents=True, exist_ok=True)

        create_caption_idx_single_scene = partial(
            self.create_caption_idx_single,
            save_path=save_path
        )

        for idx, info in tqdm.tqdm(enumerate(self.infos), total=len(self.infos)):
            create_caption_idx_single_scene((info, idx))

    def create_caption_idx_single(self, info_with_idx, save_path):
        info, idx = info_with_idx
        scene_name = info['lidar_path'].split('.')[0].replace('/', '_')
        scene_caption_save_path = save_path / (scene_name + '.pkl')

        points = self.get_lidar(info)
        data_dict = {
            'points': points
        }

        data_dict = self.get_image(info, data_dict)
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
                    create_caption_idx_single_scene, self.infos, detic_crop_infos
                ), total=len(self.infos))
            )

        with open(save_path, 'wb') as f:
            pickle.dump(detic_crop_caption_idx, f)

    def create_caption_idx_with_crop_single(self, info, detic_infos=None, window_size=None, strides=None):
        scene_name = info['lidar_path'].split('.')[0].replace('/', '_')
        if detic_infos is not None:
            assert detic_infos['scene_name'] == scene_name

        points = self.get_lidar(info)
        data_dict = {
            'points': points
        }

        scene_caption_idx = {'scene_name': scene_name, 'infos': {}, 'boxes': {}}
        data_dict = self.get_image(info, data_dict)

        # for each image
        for image_name in self.image_list:
            image_name = image_name.lower()
            if detic_infos is not None:
                image_detic_pred = detic_infos['infos'][image_name.upper()]
                boxes = image_detic_pred['boxes']
            else:
                boxes = get_sliding_windows(data_dict['image_shape'][image_name], window_size, strides)

            if boxes.shape[0] == 0:
                continue

            point_img = data_dict['point_img'][image_name]  # (x, y)
            point_img_idx = torch.from_numpy(data_dict['point_img_idx'][image_name]).int()
            # enlarge boxes with a ratio
            if args.enlarge_box_ratio > 1:
                boxes = caption_utils.enlarge_boxes_size(
                    boxes, args.enlarge_box_ratio, args.enlarge_boxes_max_thresh, data_dict['image_shape'][image_name]
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


def get_sliding_windows(image_shape, window_size, strides):
    height, width, channel = image_shape
    box_list = []  # each box should in format [y_min, x_min, y_max, x_max]

    sampling_row_coord = list(np.arange(0, height - window_size[0] - 1, strides[0]))
    sampling_col_coord = list(np.arange(0, width - window_size[1] - 1, strides[1]))

    if height - sampling_row_coord[-1] - window_size[0] > window_size[0] / 2:
        sampling_row_coord.append(height - window_size[0] - 1)

    if width - sampling_col_coord[-1] - window_size[1] > window_size[1] / 2:
        sampling_col_coord.append(width - window_size[1] - 1)

    for row in sampling_row_coord:
        for col in sampling_col_coord:
            box = (row, col, row + window_size[0], col + window_size[1])
            box_list.append(box)

    boxes = np.array(box_list)
    return boxes


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--model_cfg', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_nuscenes_caption_idx',
                        choices=['create_nuscenes_caption_idx',
                                 'create_nuscenes_caption_idx_detic_crop',
                                 'create_nuscenes_caption_idx_basic_crop',
                                 'create_nuscenes_caption_idx_detic_crop_caption'],
                        help='')

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

    # for detic crop caption
    parser.add_argument('--min_image_crop_area', default=3000, type=int, help='minimal image crop size')
    parser.add_argument('--enlarge_boxes_max_thresh', default=8000, type=int, help='maximum size that dont need a enlarge')
    parser.add_argument('--enlarge_box_ratio', default=1.0, type=float, help='enlarge the box with a ratio')

    global args
    args = parser.parse_args()

    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    dataset_cfg.VERSION = args.version

    # to load class names
    cfg = EasyDict(yaml.safe_load(open(args.model_cfg)))

    dataset = NuScenesDatasetCaption(
        dataset_cfg=dataset_cfg, class_names=cfg.CLASS_NAMES,
        root_path=ROOT_DIR / 'data' / 'nuscenes',
        training=True, logger=common_utils.create_logger()
    )
    if args.func == 'create_nuscenes_caption_idx':
        """
        python -m pcseg.datasets.nuscenes.nuscenes_caption --func create_nuscenes_caption_idx \
        --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset_multimodal.yaml \
        --model_cfg tools/cfgs/nuscenes_models/sparseunet_debug_image.yaml \
        --version v1.0-mini
        """
        dataset.create_caption_idx(args.workers)
    elif args.func == 'create_nuscenes_caption_idx_detic_crop':
        """
        python -m pcseg.datasets.nuscenes.nuscenes_caption --func create_nuscenes_caption_idx_detic_crop \
        --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset_multimodal.yaml \
        --model_cfg tools/cfgs/nuscenes_models/sparseunet_debug_image.yaml --version v1.0-mini --workers 4 \
        --detic_info_path ../Detic/nuscenes_v1.0-mini_detic_pred_results.pkl \
        --save_path data/nuscenes/v1.0-mini/nuscenes_caption_idx_detic_crop.pkl
        """
        dataset.create_caption_idx_with_crop(
            args.save_path, args.detic_info_path, num_workers=args.workers
        )
    elif args.func == 'create_nuscenes_caption_idx_basic_crop':
        """
        python -m pcseg.datasets.nuscenes.nuscenes_caption --func create_nuscenes_caption_idx_basic_crop \
        --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset_multimodal.yaml \
        --model_cfg tools/cfgs/nuscenes_models/sparseunet_debug_image.yaml --version v1.0-mini \
        --workers 16 --save_path data/nuscenes/v1.0-mini/nuscenes_caption_idx_basic_crop.pkl
        """
        dataset.create_caption_idx_with_crop(
            args.save_path, window_size=args.window_size, overlap_ratio=args.overlap_ratio, num_workers=args.workers
        )
    elif args.func == 'create_nuscenes_caption_idx_detic_crop_caption':
        """
        python -m pcseg.datasets.nuscenes.nuscenes_caption --func create_nuscenes_caption_idx_detic_crop_caption \
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
        dataset.create_caption_idx_with_crop(
            args.save_path, args.detic_info_path, num_workers=args.workers
        )
    else:
        raise NotImplementedError

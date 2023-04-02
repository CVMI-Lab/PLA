import json

import numpy as np
import tqdm
import pickle
import torch

from functools import partial
import concurrent.futures as futures

from pcseg.datasets.scannet.scannet_dataset import ScanNetDataset
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
            scene_name = info.split('/')[-1].split('.')[0]
            info = {'scene_name': scene_name, 'depth_image_size': self.dataset.depth_image_scale}
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

        self.merge_to_one_file(save_path)  # TODO

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

    def create_entity_caption_idx(self, num_workers=16):
        caption_save_path = self.dataset.root_path / 'caption_idx_{}.pickle'.format(args.tag)
        # save_path.mkdir(parents=True, exist_ok=True)

        captions_entity_corr_idx = {}
        view_caption = json.load(open(args.view_caption_path, 'r'))
        view_caption_corr_idx = pickle.load(open(args.view_caption_corr_idx_path, 'rb'))
        # res = self.model.predict_step(img_path)
        view_entity_caption = self.extract_entity(view_caption)
        captions_entity_corr_idx = self.get_entity_caption_corr_idx(
            view_entity_caption, view_caption_corr_idx
        )

        with open(caption_save_path, 'wb') as f:
            pickle.dump(captions_entity_corr_idx, f)

    @staticmethod
    def extract_entity(view_caption):
        caption_entity = {}
        for scene in view_caption:
            for frame in view_caption[scene]:
                caption = view_caption[scene][frame]
                tokens = nltk.word_tokenize(caption)
                tagged = nltk.pos_tag(tokens)
                entities = []
                # entities = nltk.chunk.ne_chunk(tagged)
                for e in tagged:
                    if e[1].startswith('NN'):
                        entities.append(e[0])
                new_caption = ' '.join(entities)
                caption_entity[scene][frame] = new_caption
        return caption_entity

    @staticmethod
    def compute_intersect_and_diff(c1, c2):
        old = set(c1) - set(c2)
        new = set(c2) - set(c1)
        intersect = set(c1) & set(c2)
        return old, new, intersect

    def get_entity_caption_corr_idx(self, view_entity_caption, view_caption_corr_idx):
        entity_caption = {}
        entity_caption_corr_idx = {}

        minpoint = 100
        ratio = args.entity_overlap_thr

        for scene in tqdm.tqdm(view_caption_corr_idx):
            assert scene in view_entity_caption
            frame_idx = view_caption_corr_idx[scene]
            entity_caption[scene] = {}
            entity_caption_corr_idx[scene] = {}
            entity_num = 0
            frame_keys = list(frame_idx.keys())
            for ii in range(len(frame_keys) - 1):
                for jj in range(ii + 1, len(frame_keys)):
                    idx1 = frame_idx[frame_keys[ii]].cpu().numpy()
                    idx2 = frame_idx[frame_keys[jj]].cpu().numpy()
                    c = view_entity_caption[scene][frame_keys[ii]].split(' ')
                    c2 = view_entity_caption[scene][frame_keys[jj]].split(' ')
                    if 'room' in c:  # remove this sweeping word
                        c.remove('room')
                    if 'room' in c2:
                        c2.remove('room')

                    old, new, intersection = self.compute_intersect_and_diff(idx1, idx2)
                    old_c, new_c, intersection_c = self.compute_intersect_and_diff(c, c2)

                    if len(intersection) > minpoint and len(intersection_c) > 0 and \
                        len(intersection) / float(min(len(idx1), len(idx2))) <= ratio:
                        entity_caption[scene]['entity_{}'.format(entity_num)] = ' '.join(list(intersection_c))
                        entity_caption_corr_idx[scene]['entity_{}'.format(entity_num)] = torch.IntTensor(list(intersection))
                        entity_num += 1
                    if len(old) > minpoint and len(old_c) > 0 and len(old) / float(len(idx1)) <= ratio:
                        entity_caption[scene]['entity_{}'.format(entity_num)] = ' '.join(list(old_c))
                        entity_caption_corr_idx[scene]['entity_{}'.format(entity_num)] = torch.IntTensor(list(old))
                        entity_num += 1
                    if len(new) > minpoint and len(new_c) > 0 and len(new) / float(len(idx2)) <= ratio:
                        entity_caption[scene]['entity_{}'.format(entity_num)] = ' '.join(list(new_c))
                        entity_caption_corr_idx[scene]['entity_{}'.format(entity_num)] = torch.IntTensor(list(new))
                        entity_num += 1

        return entity_caption_corr_idx


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

    parser.add_argument('--save_path', type=str, help='')

    # filters
    parser.add_argument('--filter_by_image_size', action='store_true', default=False, help='')
    parser.add_argument('--filter_empty_caption', action='store_true', default=False, help='')

    # entity caption
    parser.add_argument('--entity_overlap_thr', default=0.3, help='threshold ratio for filtering out large entity-level point set')
    parser.add_argument('--view_caption_path', default=None, help='path for view-level caption')
    parser.add_argument('--view_caption_corr_idx_path', default=None, help='path for view-level caption corresponding index')

    global args
    args = parser.parse_args()

    ROOT_DIR = (Path(__file__).resolve().parent / '../../').resolve()
    print(ROOT_DIR)
    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    dataset_cfg.VERSION = args.version

    # to load class names
    cfg = EasyDict(yaml.safe_load(open(args.model_cfg)))

    if args.dataset == 'scannet':
        dataset = ScanNetDataset(
            dataset_cfg=dataset_cfg, class_names=cfg.CLASS_NAMES,
            root_path=ROOT_DIR / 'data' / 'scannetv2',
            training=True, logger=common_utils.create_logger()
        )
    elif args.dataset == 's3dis':
        # TODO: to support S3DIS generate caption correponding index
        raise NotImplementedError
    else:
        raise NotImplementedError

    processor = CaptionIdxProcessor(dataset)
    if args.func == 'create_view_caption_idx':
        """
        python -m tools.process_tools.generate_caption_idx --dataset scannet --func create_view_caption_idx \
        --cfg_file tools/cfgs/dataset_configs/scannet_dataset_image.yaml \
        --model_cfg tools/cfgs/scannet_models/spconv_clip_adamw.yaml
        """
        processor.create_caption_idx(args.workers)
    elif args.func == 'create_entity_caption_idx':
        """
        python -m tools.process_tools.generate_entity_caption_idx --dataset scannet --func create_view_caption_idx \
        --cfg_file tools/cfgs/dataset_configs/scannet_dataset_image.yaml \
        --model_cfg tools/cfgs/scannet_models/spconv_clip_adamw.yaml \
        --view_caption_path ./data/scannetv2/text_embed/caption_view_scannet_vit-gpt2-image-captioning_25k.json \
        --view_caption_corr_idx_path ./data/scannetv2/scannetv2_view_vit-gpt2_matching_idx.pickle
        """
        processor.create_caption_idx(args.workers)
    else:
        raise NotImplementedError

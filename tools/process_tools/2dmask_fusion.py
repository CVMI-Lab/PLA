"""
Modified from https://github.com/facebookresearch/segment-anything

"""
import os
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import argparse
import glob
import yaml
import tqdm

from pathlib import Path
from easydict import EasyDict
from functools import partial
import concurrent.futures as futures

from pcseg.datasets.scannet.scannet_dataset import ScanNetDataset
from pcseg.utils import common_utils, caption_utils

from calc_iou_between_indices import calc_iou


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


def show_anns_v2(anns, image, reverse=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=reverse)
    final_color_mask = np.zeros(image.shape, dtype=np.float32)
    for ann in sorted_anns:
        m = np.array(ann['segmentation'])
        x_idx, y_idx = m.nonzero()
        color_mask = np.array(np.random.random((1, 3)).tolist()[0])
        final_color_mask[x_idx, y_idx] = np.array(color_mask)

    final_color_mask = np.floor(final_color_mask * 255)
    final_color_mask = np.clip(final_color_mask, a_min=0, a_max=255).astype(np.uint8)
    return final_color_mask


def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def parse_config():
    root_dir = (Path(__file__).resolve().parent / '../../').resolve()
    print(root_dir)
    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))

    # to load class names
    cfg = EasyDict(yaml.safe_load(open(args.model_cfg)))
    return root_dir, dataset_cfg, cfg


class MaskFusioner(object):
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
            xyz, rgb, _, _, _ = self.dataset.load_data(idx)
            scene_name = info.split('/')[-1].split('.')[0]
            info = {'scene_name': scene_name, 'depth_image_size': self.dataset.depth_image_scale}
            data_dict = {'points_xyz': xyz, 'rgb': rgb, 'scene_name': scene_name}
        else:
            raise NotImplementedError

        return data_dict, scene_name, info

    @staticmethod
    def get_point_image_idx(data_dict, image_name, image_shape):
        point_img = data_dict['point_img'][image_name]  # (x, y)
        if data_dict.get('depth_image_size', None):
            # rescale point image if depth image size not match real image
            scale = np.array(image_shape[:2]) / np.array(data_dict['depth_image_size'])
            point_img = point_img * scale

        point_img_idx = torch.from_numpy(data_dict['point_img_idx'][image_name]).int()
        return point_img, point_img_idx

    def fuse_mask(self, num_workers=16):
        # fuse_mask_single_func = partial(
        #     self.fuse_mask_single,
        # )

        # with futures.ThreadPoolExecutor(num_workers) as executor:
        #     final_masks = list(
        #         tqdm.tqdm(executor.map(
        #             fuse_mask_single_func, self.infos, range(len(self.infos)),
        #         ), total=len(self.infos))
        #     )

        for i in tqdm.tqdm(range(len(self.infos)), total=len(self.infos)):
            self.fuse_mask_single(self.infos[i], i)

    def fuse_mask_single(self, info, idx):
        if isinstance(info, dict):
            scene_name = info['lidar_path'].split('.')[0].replace('/', '_')
        else:
            scene_name = info.split('/')[-1].split('.')[0]

        data_dict, scene_name, info = self.get_lidar(info, idx)
        data_dict = self.dataset.get_image(info, data_dict)

        image_name_list = list(data_dict['point_img_idx'].keys())

        # 0 is not instance
        instance_mask3d_list = []
        instance_class3d_list = []
        
        if os.path.exists(os.path.join(args.output_dir, scene_name + '_3d_mask.npy')):
            print(f'{scene_name} already exists, skip')
            return

        t_bar = tqdm.tqdm(total=len(image_name_list))
        for image_name in image_name_list:
            image_name = image_name.lower()
            image_shape = data_dict['image_shape'][image_name]

            point_img, point_img_idx = self.get_point_image_idx(data_dict, image_name, image_shape)

            # get 2D instance masks
            instance_mask2d, instance_class2d = self.load_current_2d_mask(args.image_mask_path, image_name, scene_name)
            
            # project 2D instance mask to get rough 3D instance masks
            cur_instance_mask_3d = self.get_3d_instance_mask_from_2d_mask(instance_mask2d, point_img, point_img_idx)

            # merge 3D instance masks to existing ones
            if len(cur_instance_mask_3d) == 0:
                continue
            
            if len(instance_mask3d_list) > 0:
                self.merge_by_category_and_iou(
                    instance_mask3d_list, instance_class3d_list, cur_instance_mask_3d, instance_class2d
                )
            else:
                instance_mask3d_list.extend(cur_instance_mask_3d)
                instance_class3d_list.extend(instance_class2d)
                
            t_bar.update()

        # save the 3D instance mask for current scene
        self.save_3d_instance_mask_npy(data_dict['points_xyz'].shape[0], scene_name, instance_mask3d_list)

        torch.cuda.empty_cache()
        # visualize 3D instance colors
        # import tools.visual_utils.open3d_vis_utils as vis
        # inst_rgb = np.zeros(data_dict['points_xyz'].shape, dtype=np.float32)
        # point_inst_rgb = vis.get_coor_colors(global_point_inst_id)
        # inst_rgb[global_point_inst_id > 0, :] = point_inst_rgb[global_point_inst_id > 0, :]
        # import ipdb; ipdb.set_trace(context=20)
        # vis.draw_scenes(data_dict['points_xyz'], point_colors=inst_rgb, point_size=10)

    def load_current_2d_mask(self, mask_path, image_name, scene_name):
        instance_mask_path = os.path.join(mask_path, scene_name, image_name + '_instance_mask.npy')
        instance_mask = np.load(instance_mask_path)
        instance_class = np.load(instance_mask_path.replace('instance_mask', 'instance_class'))
        return instance_mask, instance_class
    
    @staticmethod
    def merge_with_propagate(global_point_inst_id, point_img_idx, local_point_inst_idx):
        # check existing fused masks
        filled_point_idx_mask = global_point_inst_id > 0
        local_check_mask = filled_point_idx_mask[point_img_idx]

        while local_check_mask.sum() > 0:
            local_check_idxs = local_check_mask.nonzero()[0]
            check_idx = local_check_idxs[0]
            global_check_idx = point_img_idx[check_idx]
            local_duplicated_mask = local_point_inst_idx == local_point_inst_idx[check_idx]
            local_point_inst_idx[local_duplicated_mask] = global_point_inst_id[global_check_idx]

            local_check_mask[local_duplicated_mask] = False

        return local_point_inst_idx

    @staticmethod
    def get_3d_instance_mask_from_2d_mask(instance_masks_2d, point_img, point_img_idx):
        instance_mask_3d_list = []
        for i in range(instance_masks_2d.shape[0]):
            cur_insatnce_mask = instance_masks_2d[i].astype(np.bool_)
            point_img_round = np.floor(point_img).astype(np.int32)
            
            # clip the value to avoid out of range
            point_img_round[:, 0] = np.clip(point_img_round[:, 0], a_min=0, a_max=instance_masks_2d.shape[2] - 1)
            point_img_round[:, 1] = np.clip(point_img_round[:, 1], a_min=0, a_max=instance_masks_2d.shape[1] - 1)
            
            instance_point_mask = cur_insatnce_mask[point_img_round[:, 1], point_img_round[:, 0]]
            instance_point_idx = point_img_idx[instance_point_mask]
            instance_mask_3d_list.append(instance_point_idx.cuda())
        
        return instance_mask_3d_list
    
    def merge_by_category_and_iou(self, scene_instance_mask3d, scene_instance_class, cur_instance_mask3d, cur_instance_class):
        iou_matrix = calc_iou(scene_instance_mask3d, cur_instance_mask3d)
        max_ious = iou_matrix.max(axis=0)
        max_ious_idx = iou_matrix.argmax(axis=0)
        
        for i in range(len(cur_instance_mask3d)):
            # iou > thresh and same category
            if max_ious[i] > args.iou_thresh and cur_instance_class[i] == scene_instance_class[max_ious_idx[i]]:
                # merge
                all_points = torch.cat([scene_instance_mask3d[max_ious_idx[i]], cur_instance_mask3d[i]], dim=0)
                unique_point_set = torch.unique(all_points)
                scene_instance_mask3d[max_ious_idx[i]] = unique_point_set
            else:
                # add new instance
                scene_instance_mask3d.append(cur_instance_mask3d[i])
                scene_instance_class.append(cur_instance_class[i])
    
    def save_3d_instance_mask_txt(self, num_points, scene_name, instance_mask3d_list):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        for i, instance_mask3d in enumerate(instance_mask3d_list):
            save_path = os.path.join(args.output_dir, scene_name + '_{:03}.txt'.format(i))
            point_mask = np.zeros(num_points, dtype=np.int32)
            point_mask[instance_mask3d.cpu()] = 1
            
            np.savetxt(save_path, point_mask, fmt='%d')
    
    def save_3d_instance_mask_npy(self, num_points, scene_name, instance_mask3d_list):
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        mask3d_cpu_list = [mask3d.cpu().numpy() for mask3d in instance_mask3d_list]
        
        np.save(
            os.path.join(args.output_dir, scene_name + '_3d_mask.npy'),
            mask3d_cpu_list
        )
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('mask2former fuse to 3D')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--model_cfg', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--workers', type=int, default=1, help='')
    
    parser.add_argument('--output_dir', type=str, default='../output/pred_v1/', help='')
    parser.add_argument('--image_mask_path', type=str, default='/home/deng/jihan/Mask2Former/output/', help='')

    parser.add_argument('--iou_thresh', type=float, default=0.1, help='merge scenes or not')
    
    global args
    args = parser.parse_args()
    print(args)

    root_dir, dataset_cfg, cfg = parse_config()

    dataset = ScanNetDataset(
        dataset_cfg=dataset_cfg, class_names=cfg.CLASS_NAMES,
        root_path=root_dir / 'data' / 'scannetv2',
        training=False, logger=common_utils.create_logger()
    )

    mask_fuser = MaskFusioner(dataset)
    mask_fuser.fuse_mask(num_workers=args.workers)

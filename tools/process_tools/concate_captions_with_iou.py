"""
This file is to merge the captions from different caption files 
into one caption file considering IoU.

"""
import torch
import json
import argparse
import tqdm
import pickle
from copy import deepcopy
import numpy as np

from calc_iou_between_indices import calc_iou


def write_caption_to_file(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)

    print(f'The caption is dump to {path}')


def replace_dict_keys_with_new_keys(origin_dict, new_key_list):
    curr_key_list = list(origin_dict.keys())
    new_dict = {}
    for i, key in enumerate(curr_key_list):
        new_dict[new_key_list[i]] = origin_dict[key]

    return new_dict


def merge_caption_idx_with_path_list(primary_caption_idx, other_caption_idx_list,
                                     primary_caption, other_caption_list):

    for i in tqdm.tqdm(range(len(primary_caption_idx))):
        if isinstance(primary_caption_idx, list):
            pri_scene_caption_idx = primary_caption_idx[i]['infos']
            scene_name = primary_caption_idx[i]['scene_name']
        elif isinstance(primary_caption_idx, dict):
            caption_idx_keys = list(primary_caption_idx.keys())
            scene_name = caption_idx_keys[i]
            pri_scene_caption_idx = primary_caption_idx[scene_name]
        else:
            raise NotImplementedError

        if args.no_cascade:
            pri_scene_caption_idx_for_iou_calc = deepcopy(pri_scene_caption_idx)
        else:
            pri_scene_caption_idx_for_iou_calc = pri_scene_caption_idx

        counter = 0
        for k, caption_idx in enumerate(other_caption_idx_list):
            if isinstance(caption_idx, list):
                assert primary_caption_idx[i]['scene_name'] == caption_idx[i]['scene_name']
                scene_caption_idx = caption_idx[i]['infos']
            elif isinstance(caption_idx, dict):
                scene_caption_idx = caption_idx[scene_name]
            else:
                raise NotImplementedError

            scene_caption_idx_values = list(scene_caption_idx.values())
            valid_mask = filter_accroding_to_iou(
                list(pri_scene_caption_idx_for_iou_calc.values()), scene_caption_idx_values,
                args.iou_high_thresh[k], args.iou_low_thresh[k])
            new_image_name_list = [f'app_{k}_{counter + j}' for j in range(valid_mask.sum())]

            # update caption idx
            new_caption_idx_list = [a_val for a_val, valid in zip(scene_caption_idx_values, valid_mask) if valid]
            new_cap_idx_dict = two_list_to_new_dict(new_image_name_list, new_caption_idx_list)

            pri_scene_caption_idx.update(new_cap_idx_dict)

            # update caption
            scene_caption_values = list(other_caption_list[k][scene_name].values())
            new_caption_list = [a_val for a_val, valid in zip(scene_caption_values, valid_mask) if valid]
            new_caption_dict = two_list_to_new_dict(new_image_name_list, new_caption_list)

            primary_caption[scene_name].update(new_caption_dict)

    with open(args.caption_idx_save_path, 'wb') as f:
        pickle.dump(primary_caption_idx, f)

    write_caption_to_file(primary_caption, args.caption_save_path)


def filter_accroding_to_iou(primary_caption_idx, second_caption_idx, iou_high_thresh, iou_low_thresh):
    iou_matrix = calc_iou(primary_caption_idx, second_caption_idx)
    ious = iou_matrix.max(axis=0)
    valid_mask_high = ious <= iou_high_thresh
    valid_mask_low = ious >= iou_low_thresh
    non_empty_mask = np.array([len(idx) > 0 for idx in second_caption_idx], dtype=np.bool_)
    valid_mask = valid_mask_high & non_empty_mask & valid_mask_low

    return valid_mask


def two_list_to_new_dict(key_list, value_list):
    res = {}
    for (key, value) in zip(key_list, value_list):
        res[key] = value

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--caption_path_list',
                        default=['data/scannetv2/text_embed/caption_dense_scannet_kosmos2_25k.json',
                                 'data/scannetv2/text_embed/caption_basic_crop_scannet_ofa_image-caption_coco_large_en_ofa_w400-500_over0.3.json'],
                        nargs='+', help='')
    parser.add_argument('--caption_idx_path_list',
                        default=['data/scannetv2/scannetv2_caption_idx_kosmos2_densecap_25k.pkl',
                                 'data/scannetv2/scannet_caption_idx_basic_crop.pkl'],
                        nargs='+', help='')
    
    parser.add_argument('--caption_save_path', required=True, type=str, help='')
    parser.add_argument('--caption_idx_save_path', required=True, type=str, help='')

    parser.add_argument('--iou_high_thresh', default=[0.2, 0.2], nargs='+', type=float, help='iou below such threshold can be merge')
    parser.add_argument('--iou_low_thresh', default=[0.0, 0.0], nargs='+', type=float, help='iou higher such threshold can be merge')

    parser.add_argument('--no_cascade', action='store_true', help='if True, the caption idx will not be cascade merged')
    
    global args
    args = parser.parse_args()
    
    print(f'caption_path_list: {args.caption_path_list}')
    print(f'caption_idx_path_list: {args.caption_idx_path_list}')
    
    caption_idx_list = []
    caption_list = []
    
    # the primary caption file is the first one in the list
    for idx, caption_idx_path in enumerate(args.caption_idx_path_list):
        # load caption idx
        caption_idx = pickle.load(open(caption_idx_path, 'rb'))
        caption_idx_list.append(caption_idx)
        # load caption
        current_caption = json.load(open(args.caption_path_list[idx], 'r'))
        caption_list.append(current_caption)
        
    primary_caption_idx = caption_idx_list[0]
    other_caption_idx_list = caption_idx_list[1:]
    
    primary_caption = caption_list[0]
    other_caption_list = caption_list[1:]
    
    merge_caption_idx_with_path_list(
        primary_caption_idx, other_caption_idx_list, primary_caption, other_caption_list
    )
    
    

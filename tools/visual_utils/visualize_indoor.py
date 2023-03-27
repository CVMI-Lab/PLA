import open3d as o3d
import argparse
import os
from operator import itemgetter
import cv2
import numpy as np
from functools import reduce
import torch
from indoor_utils.ply_utils import read_ply, write_ply


background_color = 220  # (220, 220, 220)
bg_idx = {
    'scannet': [0, 1, 19, 20],
    's3dis': [12, 13]
}


def get_input(opt):
    data_dict = {}
    if opt.dataset in ['scannet', 'scannet200']:
        input_file = os.path.join(opt.data_root, opt.split, opt.room_name + '_vh_clean_2.ply')
        label_file = os.path.join(opt.data_root, '{}_pth'.format(opt.split), opt.room_name + '.pth')

        xyz, rgb, alpha, face_indices = read_ply(input_file)
        # If there is no need to write a PLY file, the input file containing xyz and rgb information can be ignored, 
        # and this information can instead be read directly from the label_file.
        _, _, label, inst_label, *_ = torch.load(label_file)
        label[label == -100] = 20
        data_dict = {'xyz': xyz, 'rgb': rgb, 'label': label, 'inst_label': inst_label}
        data_dict['alpha'] = alpha
        data_dict['indices'] = face_indices
    elif opt.dataset == 's3dis':
        input_file = os.path.join(opt.data_root, opt.room_name + '.npy')
        data = np.load(input_file)
        xyz, rgb, label, inst_label = data[..., :3], data[..., 3:6], data[..., 6], data[..., 7]
        label[label == -100] = 13
        data_dict = {'xyz': xyz, 'rgb': rgb, 'label': label, 'inst_label': inst_label}
    return data_dict


def get_coords_color(opt):
    if opt.dataset == 'scannet':
        from indoor_utils.color_utils import SCANNET_CLASS_COLOR as CLASS_COLOR, \
            SCANNET_SEMANTIC_NAMES as SEMANTIC_NAMES, \
            SCANNET_SEMANTIC_IDX2NAME as SEMANTIC_IDX2NAME, \
            SCANNET_COLOR_DETECTRON2 as COLOR_DETECTRON2
    elif opt.dataset == 'scannet200':
        from indoor_utils.color_utils import SCANNET_CLASS_COLOR_200 as CLASS_COLOR, \
            CLASS_LABELS_200 as SEMANTIC_NAMES, \
            SCANNET_SEMANTIC_IDX2NAME as SEMANTIC_IDX2NAME, \
            SCANNET_COLOR_DETECTRON2 as COLOR_DETECTRON2
    elif opt.dataset == 's3dis':
        from indoor_utils.color_utils import S3DIS_CLASS_COLOR as CLASS_COLOR, \
            S3DIS_SEMANTIC_NAMES as SEMANTIC_NAMES, \
            S3DIS_SEMANTIC_IDX2NAME as SEMANTIC_IDX2NAME, \
            SCANNET_COLOR_DETECTRON2 as COLOR_DETECTRON2
            # S3DIS_SEMANTIC_IDXS as SEMANTIC_IDXS
    data_dict = get_input(opt)
    xyz, rgb, label, inst_label = data_dict['xyz'], data_dict['rgb'], data_dict['label'], data_dict['inst_label']

    if (opt.task == 'semantic_gt'):
        label = label.astype(int)
        label_rgb = np.zeros(rgb.shape)
        label_rgb[label >= 0] = np.array(
            itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_rgb
        if opt.ignore_bg:
            bg_mask = [label == i for i in bg_idx[opt.dataset]]
            bg_mask = reduce(lambda x, y: x | y, bg_mask)
            rgb[bg_mask] = background_color

    elif (opt.task == 'semantic_pred'):
        semantic_file = os.path.join(opt.prediction_path, 'semantic_pred', opt.room_name + '.npy')
        assert os.path.isfile(semantic_file), 'No semantic result - {}.'.format(semantic_file)
        label_pred = np.load(semantic_file).astype(int)  # 0~19
        label_pred_rgb = np.array(itemgetter(*SEMANTIC_NAMES[label_pred])(CLASS_COLOR))
        if opt.ignore_bg:
            bg_mask = [label == i for i in bg_idx[opt.dataset]]
            bg_mask = reduce(lambda x, y: x | y, bg_mask)
            label_pred_rgb[bg_mask] = background_color
        rgb = label_pred_rgb

    elif (opt.task == 'offset_semantic_pred'):
        label = label.astype(int)
        label_pred_rgb = np.zeros(rgb.shape)
        label_pred_rgb[label >= 0] = np.array(
            itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_pred_rgb

        offset_file = os.path.join(opt.prediction_path, 'offset_pred', opt.room_name + '.npy')
        assert os.path.isfile(offset_file), 'No offset result - {}.'.format(offset_file)
        offset_coords = np.load(offset_file)
        xyz[inst_label >= 0] += offset_coords[inst_label >= 0]

    elif (opt.task == 'offset_semantic_gt'):
        label = label.astype(int)
        label_rgb = np.zeros(rgb.shape)
        label_rgb[label >= 0] = np.array(
            itemgetter(*SEMANTIC_NAMES[label[label >= 0]])(CLASS_COLOR))
        rgb = label_rgb

        offset_file = os.path.join(opt.prediction_path, 'offset_label', opt.room_name + '.npy')
        assert os.path.isfile(offset_file), 'No offset result - {}.'.format(offset_file)
        offset_coords = np.load(offset_file)
        xyz[inst_label >= 0] += offset_coords[inst_label >= 0]

    # same color order according to instance pointnum
    elif (opt.task == 'instance_gt'):
        inst_label = inst_label.astype(int)
        print('Instance number: {}'.format(inst_label.max() + 1))
        inst_label_rgb = np.zeros(rgb.shape)
        ins_num = inst_label.max() + 1
        ins_pointnum = np.zeros(ins_num)
        for _ins_id in range(ins_num):
            ins_pointnum[_ins_id] = (inst_label == _ins_id).sum()
        sort_idx = np.argsort(ins_pointnum)[::-1]
        for _sort_id in range(ins_num):
            inst_label_rgb[inst_label == sort_idx[_sort_id]] = COLOR_DETECTRON2[
                _sort_id % len(COLOR_DETECTRON2)]
        rgb = inst_label_rgb

    # same color order according to instance pointnum
    elif (opt.task == 'instance_pred'):
        instance_file = os.path.join(opt.prediction_path, 'pred_instance', opt.room_name + '.txt')
        assert os.path.isfile(instance_file), 'No instance result - {}.'.format(instance_file)
        f = open(instance_file, 'r')
        masks = f.readlines()
        masks = [mask.rstrip().split() for mask in masks]
        inst_label_pred_rgb = np.ones(rgb.shape) * background_color  # grey color

        ins_num = len(masks)
        ins_pointnum = np.zeros(ins_num)
        inst_label = -100 * np.ones(rgb.shape[0]).astype(int)

        # sort score such that high score has high priority for visualization
        scores = np.array([float(x[-1]) for x in masks])
        sort_inds = np.argsort(scores)[::-1]
        for i_ in range(len(masks) - 1, -1, -1):
            i = sort_inds[i_]
            mask_path = os.path.join(opt.prediction_path, 'pred_instance', masks[i][0])
            assert os.path.isfile(mask_path), mask_path
            # if (float(masks[i][2]) < 0.09):
            #     continue
            mask = np.array(open(mask_path).read().splitlines(), dtype=int)
            print('{} {}: pointnum: {}'.format(i, masks[i], mask.sum()))
            ins_pointnum[i] = mask.sum()
            inst_label[mask == 1] = i
        sort_idx = np.argsort(ins_pointnum)[::-1]
        for _sort_id in range(ins_num):
            # if SEMANTIC_IDX2NAME[int(masks[sort_idx[_sort_id]][1])] in ['bed', 'chair', 'table', 'bookshelf', 'picture', 'sink', 'bathtub']:
            inst_label_pred_rgb[inst_label == sort_idx[_sort_id]] = COLOR_DETECTRON2[
                _sort_id % len(COLOR_DETECTRON2)]

        rgb = inst_label_pred_rgb

    data_dict['xyz'] = xyz
    data_dict['rgb'] = rgb
    return data_dict


def vis(opt):
    data_dict = get_coords_color(opt)
    if not data_dict:
        return

    if opt.out_dir is not None:
        write_ply(
            '{}/{}_{}_{}.ply'.format(opt.out_dir, opt.room_name, opt.task, opt.out_tag), data_dict
        )

    points, colors = data_dict['xyz'], data_dict['rgb'] / 255.0
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pc)

    if 'vis_list' in data_dict:
        for min_bound, max_bound, color in data_dict['vis_list']:
            box = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
            box.color = color / 255.0
            vis.add_geometry(box)
    # vis.get_render_option().point_size = 3
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', help='path to the input dataset files', default='scannet')
    parser.add_argument(
        '--data_root', help='path to the input dataset files', default='../../data/scannetv2/')
    parser.add_argument(
        '--prediction_path', help='path to the prediction results', default='./results')
    parser.add_argument('--room_name', help='room_name', default=None)
    parser.add_argument(
        '--task',
        help='input/semantic_gt/semantic_pred/offset_semantic_pred/instance_gt/instance_pred',
        default='instance_pred')
    parser.add_argument('--out_dir', default=None, help='output point cloud directory')
    parser.add_argument('--out_tag', default='', help='tag')
    parser.add_argument('--class_id', default=0, type=int, help='class id to visualize logit')
    parser.add_argument('--split', default='val', help='train / val / test')
    parser.add_argument('--ignore_bg', action='store_true', help='ignore background classes')
    # parser.add_argument('--out', default=None, help='output point cloud file in FILE.ply format')
    opt = parser.parse_args()

    if opt.out_dir is not None:
        os.makedirs(opt.out_dir, exist_ok=True)

    if opt.room_name is None:
        if opt.task == 'instance_pred':
            rooms = sorted(os.listdir(opt.prediction_path + '/pred_instance'))
        elif opt.task == 'semantic_pred':
            rooms = sorted(os.listdir(opt.prediction_path + '/semantic_pred'))
        elif opt.task == 'offset_semantic_pred':
            rooms = sorted(os.listdir(opt.prediction_path + '/offset_pred'))
        elif opt.task == 'offset_semantic_gt':
            rooms = sorted(os.listdir(opt.prediction_path + '/offset_label'))
        else:
            raise NotImplementedError

        if rooms[0] == 'predicted_masks':
            rooms = rooms[1:]
        for (i, r) in enumerate(rooms):
            print(i, r)
            opt.room_name = r.split('.')[0]
            vis(opt)
    else:
        vis(opt)

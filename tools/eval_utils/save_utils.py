import multiprocessing as mp
import os
import os.path as osp
import numpy as np
from pcseg.models.model_utils.rle_utils import rle_decode


def save_npy(root, name, scan_ids, arrs):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.npy') for i in scan_ids]
    pool = mp.Pool()
    pool.starmap(np.save, zip(paths, arrs))
    pool.close()
    pool.join()


def save_single_instance(root, scan_id, insts, nyu_id=None):
    f = open(osp.join(root, f'{scan_id}.txt'), 'w')
    os.makedirs(osp.join(root, 'predicted_masks'), exist_ok=True)
    for i, inst in enumerate(insts):
        # assert scan_id == inst['scan_id']
        label_id = inst['label_id']
        # scannet dataset use nyu_id for evaluation
        if nyu_id is not None:
            label_id = nyu_id[label_id - 1]
        conf = inst['conf']
        f.write(f'predicted_masks/{scan_id}_{i:03d}.txt {label_id} {conf:.4f}\n')
        mask_path = osp.join(root, 'predicted_masks', f'{scan_id}_{i:03d}.txt')
        mask = rle_decode(inst['pred_mask'])
        np.savetxt(mask_path, mask, fmt='%d')
    f.close()


def save_pred_instances(root, name, scan_ids, pred_insts, nyu_id=None):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)
    nyu_ids = [nyu_id] * len(scan_ids)
    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, pred_insts, nyu_ids))
    pool.close()
    pool.join()


def save_gt_instance(path, gt_inst, nyu_id=None):
    if nyu_id is not None:
        sem = gt_inst // 1000
        ignore = sem == 0
        ins = gt_inst % 1000
        nyu_id = np.array(nyu_id)
        sem = nyu_id[sem - 1]
        sem[ignore] = 0
        gt_inst = sem * 1000 + ins
    np.savetxt(path, gt_inst, fmt='%d')


def save_gt_instances(root, name, scan_ids, gt_insts, nyu_id=None):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.txt') for i in scan_ids]
    pool = mp.Pool()
    nyu_ids = [nyu_id] * len(scan_ids)
    pool.starmap(save_gt_instance, zip(paths, gt_insts, nyu_ids))
    pool.close()
    pool.join()


def save_panoptic_single(path, panoptic_pred, learning_map_inv, num_classes):
    # convert cls to kitti format
    panoptic_ids = panoptic_pred >> 16
    panoptic_cls = panoptic_pred & 0xFFFF
    new_learning_map_inv = {num_classes: 0}
    for k, v in learning_map_inv.items():
        if k == 0:
            continue
        if k < 9:
            new_k = k + 10
        else:
            new_k = k - 9
        new_learning_map_inv[new_k] = v
    panoptic_cls = np.vectorize(new_learning_map_inv.__getitem__)(panoptic_cls).astype(
        panoptic_pred.dtype)
    panoptic_pred = (panoptic_cls & 0xFFFF) | (panoptic_ids << 16)
    panoptic_pred.tofile(path)


def save_panoptic(root, name, scan_ids, arrs, learning_map_inv, num_classes):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f'{i}.label'.replace('velodyne', 'predictions')) for i in scan_ids]
    learning_map_invs = [learning_map_inv] * len(scan_ids)
    num_classes_list = [num_classes] * len(scan_ids)
    for p in paths:
        os.makedirs(osp.dirname(p), exist_ok=True)
    pool = mp.Pool()
    pool.starmap(save_panoptic_single, zip(paths, arrs, learning_map_invs, num_classes_list))

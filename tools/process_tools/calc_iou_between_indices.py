import pickle
import argparse
import numpy as np
from tqdm import tqdm
import torch


def offset_to_index(offset, empty_indices_index):
    index = torch.zeros(offset[-1].item(), dtype=torch.long).cuda()
    begin_idx = 1
    last_idx = -1
    for ii in offset[1:]:
        if ii == offset[0]:
            begin_idx += 1
        else:
            break
    for ii in torch.flip(offset[:-1], dims=[0]):
        if ii == offset[-1]:
            last_idx -= 1
        else:
            break
    index[offset[begin_idx:last_idx]] = 1
    for ii in empty_indices_index:
        if offset[ii + 1] < offset[-1]:
            index[offset[ii + 1]] += 1
    index = torch.cumsum(index, dim=0)
    return index


def calc_iou(indices1_list, indices2_list):
    indices1 = torch.cat(indices1_list, dim=0).cuda()
    indices1_offset = torch.LongTensor([0] + [len(idx) for idx in indices1_list]).cuda()
    empty_indices_index = torch.where(indices1_offset[1:] == 0)[0]
    indices1_offset = torch.cumsum(indices1_offset, 0).cuda()
    indices1_index = offset_to_index(indices1_offset, empty_indices_index).cuda()  # [K1+...+Kn,]

    indices2 = torch.cat(indices2_list, dim=0).cuda()
    indices2_offset = torch.LongTensor([0] + [len(idx) for idx in indices2_list]).cuda()
    empty_indices_index = torch.where(indices2_offset[1:] == 0)[0]
    indices2_offset = torch.cumsum(indices2_offset, 0).cuda()
    indices2_index = offset_to_index(indices2_offset, empty_indices_index).cuda()

    if len(indices1) == 0 or len(indices2) == 0:
        return np.zeros((len(indices1_list), len(indices2_list)))
    max_num_points = max(indices1.max().item(), indices2.max().item()) + 1
    indices1_onehot = torch.zeros(len(indices1_offset) - 1, max_num_points).cuda()  # [K, N]
    indices1_onehot[indices1_index, indices1.long()] = 1.  # [K, N]
    # assert indices1_onehot.sum().item() == indices1.shape[0]
    # assert indices1_onehot[-1][indices1_list[-1].long().cuda()].min().item() == 1.

    indices2_onehot = torch.zeros(len(indices2_offset) - 1, max_num_points).cuda()  # [K, N]
    indices2_onehot[indices2_index, indices2.long()] = 1.  # [K, N]
    # assert indices2_onehot.sum().item() == indices2.shape[0]
    # assert indices2_onehot[-1][indices2_list[-1].long().cuda()].min().item() == 1.

    indices_intersection = indices1_onehot @ indices2_onehot.T  # [K1, K2]
    indices_union = indices1_onehot.sum(1, keepdims=True) + indices2_onehot.sum(1, keepdims=True).T - indices_intersection
    iou = indices_intersection / (indices_union + 1e-6)
    # print('iou: ', iou.mean().item())
    return iou.cpu().numpy()


def main(args):
    with open('../data/scannetv2/scannetv2_train.txt', 'r') as f:
        train_scans = sorted(f.readlines())

    source1 = pickle.load(open(args.source1, 'rb'))
    source2 = pickle.load(open(args.source2, 'rb'))

    for ii, f in enumerate(tqdm(train_scans)):
        f = f.strip()
        print(f)

        if isinstance(source1, list):
            assert source1[ii]['scene_name'] == f.split('/')[-1]
            indices1_dict = source1[ii]['infos']
        else:
            indices1_dict = source1[f]
        indices1_list = list(indices1_dict.values())

        if isinstance(source2, list):
            indices2_dict = source2[ii]['infos']
        else:
            indices2_dict = source2[f]
        indices2_list = list(indices2_dict.values())

        # if f.split('/')[-1] != 'scene0033_00':
        #     continue

        calc_iou(indices1_list, indices2_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source1', type=str, default='source1.pkl')
    parser.add_argument('--source2', type=str, default='source2.pkl')
    args = parser.parse_args()
    main(args)
    
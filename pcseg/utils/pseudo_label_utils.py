import os
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
from functools import reduce

import torch
import torch.distributed as tdist
from pcseg.models import load_data_to_gpu
from . import common_utils, commu_utils


def generate_pseudo_label_batch(cfg, batch, train_loader, model, epoch, pseudo_labels_dir_epoch,
                                key, rank, thres, num_classes):
    # if os.path.exists(os.path.join(pseudo_labels_dir_epoch, 'txt', batch['ids'][0].split('/')[-1].split('.')[0] + '.txt'):
    #     return
    if cfg.get('GT_OFFSET', False):
        batch['gt_label_generation'] = True
    else:
        batch['pseudo_label_generation'] = True
    batch['epoch'] = epoch
    load_data_to_gpu(batch)
    if batch['points_xyz'].shape[0] < cfg.get('MAX_POINT', 8000000):
        with torch.no_grad():
            ret_dict = model(batch)
    else:
        # print(batch['ids'], batch['points_xyz'].shape[0])
        # too many points to handle, ignore
        ret_dict = {'pt_offsets': torch.zeros_like(batch['points_xyz']).cuda(), }
    torch.cuda.empty_cache()
    for k in key:
        preds = ret_dict[k].cpu()
        # output = ret['output']
        offsets = batch['offsets_all'].cpu() if 'offsets_all' in batch else batch['offsets'].cpu()
        # features = ret['feats_all'] if 'feats_all' in ret else ret['feats']
        # pseudo_labels = combine_pseudo_label_batch(cfg, batch, ret, preds)
        # pseudo_labels_2 = select_confident_label(cfg, batch, output, preds)
        # for c in range(cfg.DATA_CONFIG.DATA_CLASS.n_classes):
        #     print(c, (pseudo_labels_1 == c).sum(), diversity_measure(features[pseudo_labels_1 == c]),
        #           (pseudo_labels_2 == c).sum(), diversity_measure(features[pseudo_labels_2 == c]))

        common_utils.save_results(
            pseudo_labels_dir_epoch, preds.numpy(), offsets.cpu().numpy(), batch['ids'],
            train_loader.dataset.get_data_list(), formats=['npy'], replace=True
        )

        # compute class ratio
        class_ratio_batch = torch.histc(preds, bins=num_classes, min=0, max=num_classes - 1).cuda()
    return class_ratio_batch


def generate_pseudo_labels(cfg, logger, train_loader, model, epoch, pseudo_labels_dir_epoch, key, rank=0,
                           thres=0., done=True, dist=False, n_classes=1):
    logger.info("******************* Generating Pseudo Labels *********************")
    commu_utils.synchronize()

    if os.path.exists(pseudo_labels_dir_epoch / 'done.txt'):
        return
    model.eval()
    train_loader.dataset.set_training_mode(False)
    # train_loader.dataset.set_get_pairs(False)
    _, world_size = common_utils.get_dist_info()

    class_ratio = torch.zeros(n_classes).cuda()
    pbar = tqdm(
        total=len(train_loader) * world_size, leave=False, desc='generate pseudo labels', dynamic_ncols=True, disable= rank != 0)
    for batch in train_loader:
        class_ratio_batch = generate_pseudo_label_batch(
            cfg, batch, train_loader, model, epoch, pseudo_labels_dir_epoch, key, rank, thres, n_classes)
        if dist:
            tdist.all_reduce(class_ratio_batch)
        class_ratio += class_ratio_batch / 1000.0
        pbar.update(world_size)
    pbar.close()

    if rank == 0 and done:
        done_flag_path = pseudo_labels_dir_epoch / 'done.txt'
        np.savetxt(done_flag_path, 'done')
    commu_utils.synchronize()

    # if os.path.exists(pseudo_labels_dir_epoch.replace(str(epoch), str(epoch - args.pseudo_labels_freq))):
    #     pass
    # os.remove(pseudo_labels_dir_epoch.replace(str(epoch), str(epoch - args.pseudo_labels_freq))
    # num_workers = 8
    # with futures.ThreadPoolExecutor(num_workers) as excutor:
    #     list(tqdm(excutor.map(generate_pseudo_label_batch, train_loader)))
    train_loader.dataset.set_training_mode(True)
    return class_ratio.cpu().numpy()


def get_label_confidence(cfg, logger, train_loader, model, test_fn_st, epoch, rank, dist=False):
    logger.info("******************* Get Pseudo Label Confidence *********************")
    commu_utils.synchronize()
    if isinstance(model, list):
        for m in model:
            m.eval()
    else:
        model.eval()
    train_loader.dataset.set_training_mode(False)
    train_loader.dataset.set_get_pairs(False)

    max_points = 1000000000
    if rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train', dynamic_ncols=True)
    class_confidence = [[] for _ in range(cfg.DATA_CONFIG.DATA_CLASS.n_classes)]
    class_num = [0 for _ in range(cfg.DATA_CONFIG.DATA_CLASS.n_classes)]
    for i, batch in enumerate(train_loader):
        with torch.no_grad():
            # no seg_aggregation, no with_crop for prediction
            ret = forward_test_model(batch, model, test_fn_st, epoch)
        output = ret['output']
        pseudo_labels = ret['semantic_preds']
        output_score = torch.nn.functional.softmax(output, 1).max(1)[0]
        df = pd.DataFrame(np.stack(
            (pseudo_labels.cpu().numpy(), output_score.cpu().numpy()), 1), columns=["key", "val"])
        class_confidence_batch = dict(df.groupby("key").val.apply(pd.Series.tolist))
        class_num_batch = []
        for c in range(cfg.DATA_CONFIG.DATA_CLASS.n_classes):
            if c not in class_confidence_batch:  # fill class, avoid stuck in gather step
                class_confidence_batch[c] = []
            class_num_batch.append(len(class_confidence_batch[c]))
            random.shuffle(class_confidence_batch[c])
            if dist:
                class_confidence[c].extend(
                    reduce(lambda x, y: x + y, common_utils.all_gather_object(class_confidence_batch[c][:max_points])))
                class_num[c] += \
                    reduce(lambda x, y: x + y, common_utils.all_gather_object(class_num_batch[c])) / 1000.0
                # avoid overflow
            else:
                class_confidence[c].extend(class_confidence_batch[c][:max_points])
                class_num[c] += class_num_batch[c] / 1000.0  # avoid overflow
            class_confidence[c].sort(reverse=True)  # sort confidence
            if (i + 1) % 10 == 0:
                if len(class_confidence[c]) == 0:
                    logger.info('class: {}, thres: {}'.format(cfg.DATA_CONFIG.class_names[c], 0))
                else:
                    logger.info('class: {}, thres: {}'.format(
                        cfg.DATA_CONFIG.class_names[c], class_confidence[c][int(
                            max(1, int(cfg.SELF_TRAIN.thres_ratio[0] * len(class_confidence[int(c)]))))]))
        if rank == 0:
            pbar.update()
    if rank == 0:
        pbar.close()

    commu_utils.synchronize()
    return class_confidence, class_num


def get_thres_per_class_on_thres_ratio(cfg, logger, train_loader, model, test_fn_st, epoch, rank, dist=False):
    """given per class thres ratio, get per class thres"""
    class_confidence, _ = get_label_confidence(
        cfg, logger, train_loader, model, test_fn_st, epoch, rank, dist=dist
    )
    per_class_thres_list = []
    thres_ratio = cfg.SELF_TRAIN.thres_ratio
    if len(thres_ratio) == 1:  # global thres ratio is assigned
        thres_ratio = thres_ratio * cfg.DATA_CONFIG.DATA_CLASS.n_classes
    for c in range(len(class_confidence)):
        class_confidence[c].sort(reverse=True)
        try:
            per_class_thres_list.append(
                class_confidence[c][:int(max(1, int(thres_ratio[c] * len(class_confidence[int(c)]))))][-1])
        except IndexError:  # no point is predicted as this class
            per_class_thres_list.append(0.0)
    return per_class_thres_list


def get_thres_per_class_on_sample_ratio(cfg, logger, train_loader, model, test_fn_st, epoch, rank, dist=False):
    # TODO: re-implement
    # """given sample ratio, get per class thres"""
    class_confidence, class_num = get_label_confidence(
        cfg, logger, train_loader, model, test_fn_st, epoch, rank, dist=dist
    )
    sample_ratio = cfg.SELF_TRAIN.sample_ratio.ratio
    if len(sample_ratio) == 1:  # global thres ratio is assigned
        sample_ratio = sample_ratio * cfg.DATA_CONFIG.DATA_CLASS.n_classes
    # if rank == 0:
    #     import ipdb; ipdb.set_trace(context=10)
    base_class = cfg.SELF_TRAIN.sample_ratio.base_class  # base_class that thres=cfg.SELF_TRAIN.thres
    base_class_percent = (np.array(class_confidence[base_class]) > cfg.SELF_TRAIN.thres[0]).sum() / \
                            (len(class_confidence[base_class]) + 10e-10)
    # base_class_confidence = class_confidence[base_class_percent]
    base_class_num = class_num[base_class] * base_class_percent
    per_class_thres_list = []
    for c in range(len(class_confidence)):
        class_percent = min(1.0, sample_ratio[c] / sample_ratio[base_class] * base_class_num / (class_num[c] + 10e-10))
        try:
            class_thres = class_confidence[c][:int(max(1, int(class_percent * len(class_confidence[int(c)]))))][-1]
            per_class_thres_list.append(class_thres)
        except IndexError:
            per_class_thres_list.append(0.0)
    return per_class_thres_list


def get_perclass_thres(cfg, logger, train_loader, model, epoch, rank, key, dist=False):
    if cfg.get('GLOBAL_THRES', False):  # global threshold
        thres = cfg.THRES
    else:  # based on per class threshold
        raise NotImplementedError
        thres = get_thres_per_class_on_thres_ratio(cfg, logger, train_loader, model, epoch, rank, key, dist=dist)
    return thres


def find_pseudo_label_epoch(cur_epoch, epochs_list):
    nearest_epoch = None
    for ii in epochs_list:
        if cur_epoch >= ii:
            nearest_epoch = ii
        else:
            break
    return nearest_epoch


def generate_and_set_pseudo_labels(
    cfg, args, logger, pseudo_labels_dir, train_loader, pseudo_label_loader, model, epoch, rank=0, key=['pt_offsets'], dist=False
):
    #  generate pseudo labels
    pseudo_ep = find_pseudo_label_epoch(epoch, cfg.PSEUDO_LABEL_EPOCHS)
    pseudo_labels_dir_epoch = pseudo_labels_dir / f'epoch_{pseudo_ep}'
    commu_utils.synchronize()
    generate = False
    if not os.path.exists(pseudo_labels_dir_epoch) and pseudo_ep == epoch:
        thres = get_perclass_thres(cfg, logger, pseudo_label_loader, model, epoch, rank, key, dist=dist)
        logger.info('pseudo label threshold: {} '.format(thres))
        class_ratio = generate_pseudo_labels(
            cfg, logger, pseudo_label_loader, model, epoch, pseudo_labels_dir_epoch, key, rank, thres,
            done=False, dist=dist
        )
        class_ratio /= (class_ratio.sum() + 10e-10)
        if rank == 0:
            np.savetxt(str(pseudo_labels_dir_epoch / 'class_ratio.txt'), class_ratio)
        generate = True
    if pseudo_ep is not None:
        # set pseudo labels dir
        train_loader.dataset.set_pseudo_labels_dir(
            pseudo_labels_dir_epoch, use_pseudo_gt_label=cfg.get('USE_GT_LABEL', False)
        )
    commu_utils.synchronize()
    return generate, pseudo_labels_dir_epoch


# def load_pseudo_labels(train_loader, ids, output_dir):
#     labels = np.array([]) # .reshape(0, 1)
#     for idx in ids:
#         with open(str(output_dir / (train_loader.dataset.split + '_pseudo_labels') / 'txt' /
#                     (train_loader.dataset.data_list[idx].split('/')[-1].split('.')[0] + '.txt')), 'r') as fin:
#             _labels = np.loadtxt(fin).reshape(-1)
#             labels = np.concatenate((labels, _labels))
#     labels = torch.from_numpy(labels).long()
#     return labels

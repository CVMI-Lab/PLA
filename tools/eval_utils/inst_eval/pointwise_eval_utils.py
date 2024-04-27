import numpy as np


def evaluate_semantic_acc(pred_list, gt_list, ignore_label=-100, logger=None):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    assert gt.shape == pred.shape
    correct = (gt[gt != ignore_label] == pred[gt != ignore_label]).sum()
    whole = (gt != ignore_label).sum()
    acc = correct.astype(float) / whole * 100
    logger.info(f'Acc: {acc:.1f}')
    return acc


def evaluate_semantic_miou(n_classes, pred_list, gt_list, ignore_label=-100, logger=None):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    pos_inds = gt != ignore_label
    gt = gt[pos_inds]
    pred = pred[pos_inds]
    assert gt.shape == pred.shape
    iou_list = []
    for _index in range(n_classes):
        if _index != ignore_label:
            intersection = ((gt == _index) & (pred == _index)).sum()
            union = ((gt == _index) | (pred == _index)).sum()
            iou = intersection.astype(float) / (union + 1e-10) * 100
            iou_list.append(iou)
    miou = np.nanmean(iou_list)
    logger.info('Class-wise mIoU: ' + ' '.join(f'{x:.1f}' for x in iou_list))
    logger.info(f'mIoU: {miou:.1f}')
    return miou, iou_list


def evaluate_offset_mae(pred_list, gt_list, gt_instance_list, ignore_label=-100, logger=None):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    gt_instance = np.concatenate(gt_instance_list, axis=0)
    pos_inds = gt_instance != ignore_label
    gt = gt[pos_inds]
    pred = pred[pos_inds]
    mae = np.abs(gt - pred).sum() / pos_inds.sum()
    logger.info(f'Offset MAE: {mae:.3f}')
    return mae


def evaluate_offset_mae_class(pred_list, gt_list, gt_sem_list, gt_instance_list, n_classes, ignore_label=-100, logger=None):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    gt_sem = np.concatenate(gt_sem_list, axis=0)
    gt_instance = np.concatenate(gt_instance_list, axis=0)
    pos_inds = gt_instance != ignore_label
    gt = gt[pos_inds]
    pred = pred[pos_inds]
    sem = gt_sem[pos_inds]

    mae_list = []
    for ii in range(n_classes):
        if (sem == ii).sum().item() == 0:
            ae = np.nan
        else:
            ae = np.abs(gt[sem == ii] - pred[sem == ii]).sum() / (sem == ii).sum()
        mae_list.append(ae)
    logger.info('Class Offset MAE: {}'.format(' '.join('{:.2f}'.format(ae) for ae in mae_list)))

    return np.nanmean(mae_list), mae_list

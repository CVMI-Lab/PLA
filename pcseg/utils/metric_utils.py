import numpy as np


def get_open_vocab_metric(metric_class, base_class_idx, novel_class_idx):
    if isinstance(metric_class, list):
        metric_class = np.array(metric_class)
    metric_base = np.mean(metric_class[base_class_idx])
    metric_novel = np.mean(metric_class[novel_class_idx])
    h_metric = 2 * metric_base * metric_novel / (metric_base + metric_novel + 10e-10)
    m_metric = (metric_base * len(base_class_idx) + metric_novel * len(novel_class_idx)) / (len(base_class_idx) + len(novel_class_idx))
    return h_metric, m_metric, metric_base, metric_novel


def cal_ov_metrics(cfg, logger, class_names, iou_class, acc_class, binary_acc_class):
    base_class_idx = cfg.DATA_CONFIG.base_class_idx
    novel_class_idx = cfg.DATA_CONFIG.novel_class_idx
    if cfg.DATA_CONFIG.get('trainonly_class_idx', None):
        trainonly_class_idx = cfg.DATA_CONFIG.trainonly_class_idx
        base_class_idx = [idx for idx in base_class_idx if idx not in trainonly_class_idx]
        novel_class_idx = [idx for idx in novel_class_idx if idx not in trainonly_class_idx]

    logger.info('----------- base class -----------')
    for i in base_class_idx:
        logger.info('Class {} : iou/acc/b_acc {:.4f}/{:.4f}/{:.4f}.'.format(
            class_names[i], iou_class[i], acc_class[i], binary_acc_class[i])
        )
    logger.info('----------- novel class -----------')
    for i in novel_class_idx:
        logger.info('Class {} : iou/acc/b_acc {:.4f}/{:.4f}/{:.4f}.'.format(
            class_names[i], iou_class[i], acc_class[i], binary_acc_class[i])
        )
    hiou, miou, iou_base, iou_novel = get_open_vocab_metric(
        iou_class, base_class_idx, novel_class_idx
    )
    hacc, macc, acc_base, acc_novel = get_open_vocab_metric(
        acc_class, base_class_idx, novel_class_idx
    )
    return hiou, miou, iou_base, iou_novel, hacc, macc, acc_base, acc_novel

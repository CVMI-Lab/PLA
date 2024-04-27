# Adapt from https://github.com/PRBonn/semantic-kitti-api/blob/master/auxiliary/eval_np.py
import multiprocessing as mp

import numpy as np


class PanopticEval:

    def __init__(self, dataset, offset=2**32, min_points=50, ignore_label=-100):
        self.valid_class_idx = dataset.valid_class_idx
        self.thing_class_idx = dataset.inst_class_idx
        self.stuff_class_idx = dataset.stuff_class_idx
        self.class_names = dataset.class_names
        self.n_classes = len(self.class_names)
        self.ignore_label = ignore_label
        self.offset = offset  # largest number of instances in a given scan
        self.min_points = min_points  # smallest number of points to consider instances in gt
        self.eps = 1e-15

    def evaluate_single(self, panoptic_pred, y_sem_row, y_inst_row):
        # panoptic vars
        pan_tp = np.zeros(self.n_classes, dtype=np.int64)
        pan_iou = np.zeros(self.n_classes, dtype=np.double)
        pan_fp = np.zeros(self.n_classes, dtype=np.int64)
        pan_fn = np.zeros(self.n_classes, dtype=np.int64)

        # semantic vars
        seen = np.zeros(self.n_classes, dtype=np.int64)
        correct = np.zeros(self.n_classes, dtype=np.int64)
        positive = np.zeros(self.n_classes, dtype=np.int64)

        x_sem_row = panoptic_pred & 0xFFFF
        x_inst_row = panoptic_pred

        # convert instance label: stuff -> 0, thing 1->N
        y_inst_row[y_inst_row == self.ignore_label] = -1
        y_inst_row = y_inst_row + 1

        # make sure instances are not zeros (it messes with my approach)
        x_inst_row = x_inst_row + 1
        y_inst_row = y_inst_row + 1

        # only interested in points that are outside the void area (not in excluded classes)
        gt_not_in_excl_mask = y_sem_row != self.ignore_label
        x_sem_row = x_sem_row[gt_not_in_excl_mask]
        y_sem_row = y_sem_row[gt_not_in_excl_mask]
        x_inst_row = x_inst_row[gt_not_in_excl_mask]
        y_inst_row = y_inst_row[gt_not_in_excl_mask]

        # semantic eval
        for cl in range(self.n_classes):
            seen[cl] = (y_sem_row == cl).sum()
            correct[cl] = ((y_sem_row == cl) & (x_sem_row == cl)).sum()
            positive[cl] = (x_sem_row == cl).sum()

        # panoptic eval
        # first step is to count intersections > 0.5 IoU for each class (except the ignored ones)
        for cl in range(self.n_classes):
            # get a class mask
            x_inst_in_cl_mask = x_sem_row == cl
            y_inst_in_cl_mask = y_sem_row == cl

            # get instance points in class (makes outside stuff 0)
            x_inst_in_cl = x_inst_row * x_inst_in_cl_mask.astype(np.int64)
            y_inst_in_cl = y_inst_row * y_inst_in_cl_mask.astype(np.int64)

            # generate the areas for each unique instance prediction
            unique_pred, counts_pred = np.unique(x_inst_in_cl[x_inst_in_cl > 0], return_counts=True)
            id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
            matched_pred = np.array([False] * unique_pred.shape[0])
            # print("Unique predictions:", unique_pred)

            # generate the areas for each unique instance gt_np
            unique_gt, counts_gt = np.unique(y_inst_in_cl[y_inst_in_cl > 0], return_counts=True)
            id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
            matched_gt = np.array([False] * unique_gt.shape[0])
            # print("Unique ground truth:", unique_gt)

            # generate intersection using offset
            valid_combos = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
            offset_combo = x_inst_in_cl[valid_combos] + self.offset * y_inst_in_cl[valid_combos]
            unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

            # generate an intersection map
            # count the intersections with over 0.5 IoU as TP
            gt_labels = unique_combo // self.offset
            pred_labels = unique_combo % self.offset
            gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
            pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])
            intersections = counts_combo
            unions = gt_areas + pred_areas - intersections
            ious = intersections.astype(np.float) / unions.astype(np.float)

            tp_indexes = ious > 0.5
            pan_tp[cl] += np.sum(tp_indexes)
            pan_iou[cl] += np.sum(ious[tp_indexes])

            matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
            matched_pred[[id2idx_pred[id] for id in pred_labels[tp_indexes]]] = True

            # count the FN
            pan_fn[cl] += np.sum(np.logical_and(counts_gt >= self.min_points,
                                                matched_gt == False))  # noqa

            # count the FP
            pan_fp[cl] += np.sum(
                np.logical_and(counts_pred >= self.min_points, matched_pred == False))  # noqa
        return pan_tp, pan_iou, pan_fp, pan_fn, seen, correct, positive

    def getPQ(self):
        # first calculate for all classes
        sq_all = self.pan_iou.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double), self.eps)
        rq_all = self.pan_tp.astype(np.double) / np.maximum(
            self.pan_tp.astype(np.double) + 0.5 * self.pan_fp.astype(np.double) +
            0.5 * self.pan_fn.astype(np.double), self.eps)
        pq_all = sq_all * rq_all

        # then do the REAL mean (no ignored classes)
        SQ = sq_all[self.include].mean()
        RQ = rq_all[self.include].mean()
        PQ = pq_all[self.include].mean()

        return PQ, SQ, RQ, pq_all, sq_all, rq_all

    def evaluate(self, panoptic_preds, sem_labels, inst_labels, dataset, logger):
        pool = mp.Pool()
        results = pool.starmap(self.evaluate_single, zip(panoptic_preds, sem_labels, inst_labels))
        pool.close()
        pool.join()
        # for debug
        # results = [self.evaluate_single(panoptic_preds[0], sem_labels[0], inst_labels[0])]

        pan_tp, pan_iou, pan_fp, pan_fn, seen, correct, positive = list(zip(*results))
        pan_tp = np.stack(pan_tp).sum(axis=0)
        pan_iou = np.stack(pan_iou).sum(axis=0)
        pan_fp = np.stack(pan_fp).sum(axis=0)
        pan_fn = np.stack(pan_fn).sum(axis=0)
        seen = np.stack(seen).sum(axis=0)
        correct = np.stack(correct).sum(axis=0)
        positive = np.stack(positive).sum(axis=0)

        iou_all = correct / np.maximum((seen + positive - correct).astype(np.double), self.eps)
        sq_all = pan_iou.astype(np.double) / np.maximum(pan_tp.astype(np.double), self.eps)
        rq_all = pan_tp.astype(np.double) / np.maximum(
            pan_tp.astype(np.double) + 0.5 * pan_fp.astype(np.double) +
            0.5 * pan_fn.astype(np.double), self.eps)
        pq_all = sq_all * rq_all
        pq_dagger_all = pq_all.copy()
        pq_dagger_all[self.stuff_class_idx] = iou_all[self.stuff_class_idx]

        pq_all *= 100
        sq_all *= 100
        rq_all *= 100
        iou_all *= 100
        pq_dagger_all *= 100
        SQ = sq_all.mean()
        RQ = rq_all.mean()
        PQ = pq_all.mean()
        PQ_dagger = pq_dagger_all.mean()
        IoU = iou_all.mean()
        self.print_results(
            PQ, PQ_dagger, SQ, RQ, IoU, pq_all, pq_dagger_all, sq_all, rq_all,
            iou_all, dataset, logger)
        return PQ, PQ_dagger, SQ, RQ, IoU, pq_all, pq_dagger_all, sq_all, rq_all, iou_all

    def getline(self, label_name, metric_list):
        sep     = ""
        col1    = ":"
        line  = "{:<15}".format(label_name) + sep + col1
        for m in metric_list:
            line += sep + "{:>7.2f}".format(m ) + sep
        return line

    def print_results(self, PQ, PQ_dagger, SQ, RQ, IoU, pq_all, pq_dagger_all, sq_all, rq_all,
                      iou_all, dataset, logger):
        # n_stuff = len(self.stuff_classes)
        # n_thing = len(self.thing_classes)
        pq_stuff = np.full(pq_all.shape, np.nan)
        sq_stuff = np.full(pq_all.shape, np.nan)
        rq_stuff = np.full(pq_all.shape, np.nan)
        pq_thing = np.full(pq_all.shape, np.nan)
        rq_thing = np.full(pq_all.shape, np.nan)
        sq_thing = np.full(pq_all.shape, np.nan)

        # pq_stuff[:n_stuff] = pq_all[:n_stuff]
        # sq_stuff[:n_stuff] = sq_all[:n_stuff]
        # rq_stuff[:n_stuff] = rq_all[:n_stuff]
        # pq_thing[-n_thing:] = pq_all[-n_thing:]
        # sq_thing[-n_thing:] = sq_all[-n_thing:]
        # rq_thing[-n_thing:] = rq_all[-n_thing:]
        pq_stuff[self.stuff_class_idx] = pq_all[self.stuff_class_idx]
        sq_stuff[self.stuff_class_idx] = sq_all[self.stuff_class_idx]
        rq_stuff[self.stuff_class_idx] = rq_all[self.stuff_class_idx]
        pq_thing[self.thing_class_idx] = pq_all[self.thing_class_idx]
        sq_thing[self.thing_class_idx] = sq_all[self.thing_class_idx]
        rq_thing[self.thing_class_idx] = rq_all[self.thing_class_idx]

        sep = ''
        col1 = ':'
        lineLen = 81

        logger.info('#' * lineLen)
        line = ''
        line += '{:<14}'.format('what') + sep + col1
        line += '{:>7}'.format('PQ') + sep
        line += '{:>7}'.format('PQ*') + sep
        line += '{:>7}'.format('RQ') + sep
        line += '{:>7}'.format('SQ') + sep
        line += '{:>7}'.format('PQ_t') + sep
        line += '{:>7}'.format('RQ_t') + sep
        line += '{:>7}'.format('SQ_t') + sep
        line += '{:>7}'.format('PQ_s') + sep
        line += '{:>7}'.format('RQ_s') + sep
        line += '{:>7}'.format('SQ_s') + sep
        line += '{:>7}'.format('mIoU') + sep
        logger.info(line)
        logger.info('#' * lineLen)

        if hasattr(dataset, 'base_inst_class_idx'):
            base_class_idx = dataset.base_class_idx
            novel_class_idx = dataset.novel_class_idx
            # base_inst_class_idx = dataset.base_inst_class_idx
            # novel_inst_class_idx = dataset.novel_inst_class_idx
            # ====== base ======
            metrics = [[] for _ in range(11)]
            for i in base_class_idx:
                metric = [pq_all[i], pq_dagger_all[i], rq_all[i], sq_all[i], \
                          pq_thing[i], rq_thing[i], sq_thing[i], \
                          pq_stuff[i], rq_stuff[i], sq_stuff[i], iou_all[i]]
                line = self.getline(self.class_names[i], metric)
                logger.info(line)
                for j in range(11):
                    metrics[j].append(metric[j])
            logger.info("-" * lineLen)
            line = self.getline('base_average', [np.nanmean(i) for i in metrics])
            logger.info(line)
            logger.info("=" * lineLen)
            # ====== novel ======
            metrics = [[] for _ in range(11)]
            for i in novel_class_idx:
                metric = [pq_all[i], pq_dagger_all[i], rq_all[i], sq_all[i], \
                          pq_thing[i], rq_thing[i], sq_thing[i], \
                          pq_stuff[i], rq_stuff[i], sq_stuff[i], iou_all[i]]
                line = self.getline(self.class_names[i], metric)
                logger.info(line)
                for j in range(11):
                    metrics[j].append(metric[j])
            logger.info("-" * lineLen)
            line = self.getline('novel_average', [np.nanmean(i) for i in metrics])
            logger.info(line)

        else:
            for i in self.valid_class_idx:
                line = self.getline(
                    self.class_names[i],
                    [pq_all[i], pq_dagger_all[i], rq_all[i], sq_all[i], \
                     pq_thing[i], rq_thing[i], sq_thing[i], \
                     pq_stuff[i], rq_stuff[i], sq_stuff[i], iou_all[i]])
                logger.info(line)

        logger.info('-' * lineLen)
        line = self.getline(
            'average', [PQ, PQ_dagger, RQ, SQ, \
                np.nanmean(pq_thing[self.valid_class_idx]), np.nanmean(rq_thing[self.valid_class_idx]),
                np.nanmean(sq_thing[self.valid_class_idx]), np.nanmean(pq_stuff[self.valid_class_idx]),
                np.nanmean(rq_stuff[self.valid_class_idx]), np.nanmean(sq_stuff[self.valid_class_idx]), IoU])
        logger.info(line)
        logger.info('#' * lineLen)

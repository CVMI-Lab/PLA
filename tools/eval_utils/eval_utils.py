import numpy as np
import torch
import tqdm
from functools import reduce

from pcseg.models import load_data_to_gpu
from pcseg.utils import common_utils, commu_utils, metric_utils
from .inst_eval.eval_utils import ScanNetEval
from .inst_eval.pointwise_eval_utils import evaluate_semantic_miou, evaluate_semantic_acc, evaluate_offset_mae
from .save_utils import save_npy, save_pred_instances


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, writer=None,
                   best_metric=0.0, best_epoch=-1, task='sem', eval_output_dir=None):
    if task == 'sem':
        return eval_sem(
            cfg, args, model, dataloader, epoch_id, logger, dist_test, writer, best_metric,
            best_epoch, eval_output_dir)
    elif task == 'inst':
        return eval_inst(
            cfg, args, model, dataloader, epoch_id, logger, dist_test, writer, best_metric,
            best_epoch, eval_output_dir)


def eval_sem(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, writer=None,
             best_metric=0.0, best_epoch=-1, eval_output_dir=None):
    world_size = commu_utils.get_world_size()
    dataset = dataloader.dataset

    class_names = dataset.class_names
    num_class = len(dataset.class_names)
    dataset.set_class_mode('all')

    intersection_meter = common_utils.AverageMeter()
    union_meter = common_utils.AverageMeter()
    target_meter = common_utils.AverageMeter()
    output_meter = common_utils.AverageMeter()
    binary_intersection_meter = common_utils.AverageMeter()
    binary_target_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)

    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        batch_dict['epoch'] = epoch_id - 1
        with torch.no_grad():
            ret_dict = model(batch_dict)
        preds, labels = ret_dict['seg_preds'], ret_dict['seg_labels']
        disp_dict = {}

        cur_sample_id = (i * args.test_batch_size + (
            len(batch_dict['ids']) - 1)) * world_size + cfg.LOCAL_RANK + 1
        if dist_test and cur_sample_id > len(dataset):
            preds, labels = preds[:batch_dict['offsets'][-2]], labels[:batch_dict['offsets'][-2]]
            if cfg.MODEL.get('BINARY_HEAD', False):
                ret_dict['binary_preds'] = ret_dict['binary_preds'][:batch_dict['offsets'][-2]]
            batch_dict['offsets'] = batch_dict['offsets'][:-1]
            batch_dict['ids'] = batch_dict['ids'][:-1]

        # calculate metric
        intersection_meter, union_meter, target_meter, output_meter, _ = common_utils.update_meter(
            intersection_meter, union_meter, target_meter, output_meter, preds, labels,
            num_class, ignore_label=cfg.DATA_CONFIG.get('IGNORE_LABEL', 255)
        )
        if cfg.MODEL.get('BINARY_HEAD', False):
            binary_preds = ret_dict['binary_preds']
            binary_intersection_meter, binary_target_meter = common_utils.update_binary_acc_meter(
                binary_intersection_meter, binary_target_meter, binary_preds, labels,
                cfg.DATA_CONFIG.novel_class_idx, num_class
            )

        # ==== save to file ====
        if hasattr(args, 'save_results') and len(args.save_results) > 0:
            if 'semantic' in args.save_results:
                sem_preds = preds.clone().cpu().numpy()
                pred, scene_names = [], []
                for ii in range(len(batch_dict['offsets']) - 1):
                    scene_names.append(batch_dict['scene_name'][ii])
                    pred.append(sem_preds[batch_dict['offsets'][ii]: batch_dict['offsets'][ii + 1]])
                save_npy(eval_output_dir, 'semantic_pred', scene_names, pred)
            if 'logit' in args.save_results:
                logit = ret_dict['seg_scores'].cpu().numpy()
                pred, scene_names = [], []
                for ii in range(len(batch_dict['offsets']) - 1):
                    scene_names.append(batch_dict['scene_name'][ii])
                    pred.append(logit[batch_dict['offsets'][ii]: batch_dict['offsets'][ii + 1]])
                save_npy(eval_output_dir, 'logit', scene_names, pred)
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    _, _, _, allPre, allAcc, iou_class, precision_class, acc_class, _ = \
        common_utils.calc_metrics(intersection_meter, union_meter, target_meter, output_meter)

    if cfg.MODEL.get('BINARY_HEAD', False):
        binary_macc, binary_all_acc, binary_acc_class = common_utils.calc_binary_acc(
            binary_intersection_meter, binary_target_meter
        )
    else:
        binary_macc, binary_all_acc = 0.0, 0.0
        binary_acc_class = np.zeros(acc_class.shape)

    # logger.info('Val result: mIoU/mPre/mAcc/allPre/allAcc/b_mAcc/b_allAcc \
    #         {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(
    #     mIoU, mPre, mAcc, allPre, allAcc, binary_macc, binary_all_acc))

    if 'base_class_idx' in cfg.DATA_CONFIG:
        hiou, miou, iou_base, iou_novel, hacc, macc, acc_base, acc_novel = metric_utils.cal_ov_metrics(
            cfg, logger, class_names, iou_class, acc_class, binary_acc_class
        )
        logger.info('-----------------------------------')
        logger.info('hIoU/mIoU/IoU_base/IoU_novel: {:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(hiou, miou, iou_base, iou_novel))
        logger.info('hAcc/mAcc/Acc_base/Acc_novel: {:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(hacc, macc, acc_base, acc_novel))
        metric = hiou
        if cfg.MODEL.get('BINARY_HEAD', False):
            logger.info('binary_mAcc/binary_allAcc: {:.4f}/{:.4f}.'.format(binary_macc, binary_all_acc))
    else:
        for i in dataloader.dataset.valid_class_idx:
            logger.info('Class {} : iou/acc/b_acc {:.4f}/{:.4f}/{:.4f}.'.format(
                class_names[i], iou_class[i], acc_class[i], binary_acc_class[i])
            )
        miou = np.mean(np.array(iou_class)[dataloader.dataset.valid_class_idx])
        macc = np.mean(np.array(acc_class)[dataloader.dataset.valid_class_idx])
        logger.info('-----------------------------------')
        logger.info('mIoU: {:.4f}'.format(miou)) 
        logger.info('mAcc: {:.4f}'.format(macc)) 
        metric = miou

    if writer is not None and cfg.LOCAL_RANK == 0:
        writer.add_scalar('mIoU_val', miou, epoch_id + 1)
        writer.add_scalar('allAcc_val', allAcc, epoch_id + 1)
        writer.add_scalar('binary_mAcc_val', binary_macc, epoch_id + 1)
        writer.add_scalar('binary_allAcc_val', binary_all_acc, epoch_id + 1)
        if 'base_class_idx' in cfg.DATA_CONFIG:
            writer.add_scalar('hIoU_val', hiou, epoch_id + 1)
            writer.add_scalar('IoU_novel_val', iou_novel, epoch_id + 1)
    torch.cuda.empty_cache()
    logger.info('****************Evaluation done.*****************')
    if best_metric is not None:
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch_id

        logger.info('Best epoch: {}, best metric: {}'.format(best_epoch, best_metric))
    return best_metric, best_epoch


def eval_inst(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, writer=None,
              best_metric=None, best_epoch=-1, eval_output_dir=None):
    # results = []
    scan_ids, coords, colors = [], [], []
    all_sem_preds, all_sem_labels, all_offset_preds, all_offset_labels = [], [], [], []
    all_inst_labels, all_pred_insts, all_gt_insts = [], [], []
    world_size = commu_utils.get_world_size()
    dataset = dataloader.dataset
    dataloader.dataset.set_class_mode('all')
    # class_names = dataset.class_names
    num_class = len(dataset.class_names)

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    model.eval()

    semantic_only = epoch_id - 1 < cfg.MODEL.INST_HEAD.CLUSTERING.PREPARE_EPOCH

    progress_bar = tqdm.tqdm(total=len(dataloader) * world_size, disable=cfg.LOCAL_RANK != 0)
    for i, batch in enumerate(dataloader):
        load_data_to_gpu(batch)
        batch['epoch'] = epoch_id - 1
        with torch.no_grad():
            ret_dict = model(batch)
        disp_dict = {}

        cur_sample_id = (i * args.test_batch_size) * world_size + cfg.LOCAL_RANK + 1
        if dist_test and cur_sample_id > dataloader.dataset.__len__():
            continue
        scan_ids.append(batch['scene_name'][0])
        coords.append(batch['points_xyz'])
        colors.append(batch['rgb'])
        all_sem_preds.append(ret_dict['seg_preds'].cpu().numpy())
        all_sem_labels.append(ret_dict['seg_labels'].cpu().numpy())
        all_offset_preds.append(ret_dict['pt_offsets'].cpu().numpy())
        all_offset_labels.append(ret_dict['pt_offset_label'].cpu().numpy())
        all_inst_labels.append(ret_dict['inst_label'].cpu().numpy())
        if not semantic_only:
            all_pred_insts.append(ret_dict['pred_instances'])
            all_gt_insts.append(ret_dict['gt_instances'])

        # results.append(result)
        progress_bar.set_postfix(disp_dict)
        progress_bar.update(world_size)
    progress_bar.close()
    # results = common_utils.collect_results_gpu(results, len(dataset))
    if dist_test:
        all_sem_preds = reduce(lambda x, y: x + y, commu_utils.all_gather(all_sem_preds))
        all_sem_labels = reduce(lambda x, y: x + y, commu_utils.all_gather(all_sem_labels))
        all_offset_preds = reduce(lambda x, y: x + y, commu_utils.all_gather(all_offset_preds))
        all_offset_labels = reduce(lambda x, y: x + y, commu_utils.all_gather(all_offset_labels))
        all_inst_labels = reduce(lambda x, y: x + y, commu_utils.all_gather(all_inst_labels))
        if not semantic_only:
            all_pred_insts = reduce(lambda x, y: x + y, commu_utils.all_gather(all_pred_insts))
            all_gt_insts = reduce(lambda x, y: x + y, commu_utils.all_gather(all_gt_insts))
    if cfg.LOCAL_RANK == 0:
        logger.info('Evaluate semantic segmentation and offset MAE')
        ignore_label = cfg.DATA_CONFIG.IGNORE_LABEL
        miou, iou_list = evaluate_semantic_miou(
            num_class, all_sem_preds, all_sem_labels, ignore_label, logger
        )
        acc = evaluate_semantic_acc(
            all_sem_preds, all_sem_labels, ignore_label, logger
        )
        mae = evaluate_offset_mae(
            all_offset_preds, all_offset_labels, all_inst_labels, ignore_label, logger
        )

        if not semantic_only:
            logger.info('Evaluate instance segmentation')
            # import ipdb; ipdb.set_trace(context=10)
            eval_min_npoint = getattr(cfg.MODEL.INST_HEAD, 'EVAL_MIN_POINT', None)
            inst_class_idx = dataloader.dataset.inst_class_idx
            inst_class_names = (np.array(dataset.class_names)[inst_class_idx]).tolist()
            scannet_eval = ScanNetEval(inst_class_idx, inst_class_names, eval_min_npoint)
            eval_res = scannet_eval.evaluate(all_pred_insts, all_gt_insts)
            scannet_eval.print_results(eval_res, logger, iou_list, dataset)
            if writer is not None:
                writer.add_scalar('val/AP', eval_res['all_ap'], epoch_id)
                writer.add_scalar('val/AP_50', eval_res['all_ap_50%'], epoch_id)
                writer.add_scalar('val/AP_25', eval_res['all_ap_25%'], epoch_id)
            logger.info('AP: {:.3f}. AP_50: {:.3f}. AP_25: {:.3f}'.format(
                eval_res['all_ap'], eval_res['all_ap_50%'], eval_res['all_ap_25%']))
            if hasattr(dataset, 'base_inst_class_idx'):
                res = []
                for c in eval_res['classes']:
                    res.append(eval_res['classes'][c]['ap50%'] * 100.0)
                hAP, mAP, AP_base, AP_novel = metric_utils.get_open_vocab_metric(res,
                    dataset.base_inst_class_idx, dataset.novel_inst_class_idx)
                logger.info('hAP/mAP/AP_base/AP_novel (50%): {:.2f}/{:.2f}/{:.2f}/{:.2f}'.format(hAP, mAP, AP_base, AP_novel))
                _AP = hAP
            else:
                _AP = eval_res['all_ap_50%']
        else:
            _AP = 0
        if writer is not None:
            writer.add_scalar('val/mIoU', miou, epoch_id)
            writer.add_scalar('val/Acc', acc, epoch_id)
            writer.add_scalar('val/Offset MAE', mae, epoch_id)
        if hasattr(dataset, 'base_inst_class_idx'):
            hiou, miou, iou_base, iou_novel = metric_utils.get_open_vocab_metric(
                iou_list, list(set(dataset.base_class_idx) & set(dataset.inst_class_idx)),
                dataset.novel_class_idx
            )
            logger.info('hIoU/mIoU/IoU_base/IoU_novel: {:.2f}/{:.2f}/{:.2f}/{:.2f}'.format(
                hiou, miou, iou_base, iou_novel))
            _iou = hiou
        else:
            _iou = miou
        metric = _AP
        logger.info('****************Evaluation done.*****************')
        if best_metric is not None:
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch_id

            logger.info('Best epoch: {}, best metric: {}'.format(best_epoch, best_metric))

        # ======== save to file =============
        if hasattr(args, 'save_results') and len(args.save_results) > 0:
            logger.info('Save results ...')
            if 'coords' in args.save_results:
                save_npy(eval_output_dir, 'coords', scan_ids, coords)
                save_npy(eval_output_dir, 'colors', scan_ids, colors)
            if 'semantic' in args.save_results:
                save_npy(eval_output_dir, 'semantic_pred', scan_ids, all_sem_preds)
                # save_npy(eval_output_dir, 'semantic_label', scan_ids, sem_labels)
            if 'offset' in args.save_results:
                save_npy(eval_output_dir, 'offset_pred', scan_ids, all_offset_preds)
                save_npy(eval_output_dir, 'offset_label', scan_ids, all_offset_labels)
            if 'instance' in args.save_results:
                nyu_id = dataloader.dataset.NYU_ID
                save_pred_instances(eval_output_dir, 'pred_instance', scan_ids, all_pred_insts, nyu_id)
        torch.cuda.empty_cache()
        return best_metric, best_epoch
    else:
        return None, None


if __name__ == '__main__':
    pass

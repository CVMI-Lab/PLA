import _init_path
import argparse
import datetime
import glob
import os
from pathlib import Path
import subprocess

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcseg.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcseg.datasets import build_dataloader
from pcseg.models import build_vision_network, build_text_network
from pcseg.models.text_networks import load_text_embedding_from_encoder, load_text_embedding_from_path
from pcseg.models.model_utils import load_best_metric
from pcseg.utils import common_utils, caption_utils, commu_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model

import warnings
warnings.filterwarnings("ignore")


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distributed training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=5, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=2, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    parser.add_argument('--print_freq', type=int, default=5, help='')
    parser.add_argument('--eval_freq', type=int, default=10, help='evaluation frequency during training')
    parser.add_argument('--validate_start', action='store_true', default=False, help='evaluation at the begining')
    parser.add_argument('--use_amp', action='store_true', default=False, help='')
    parser.add_argument('--multi_epoch_loader', action='store_true', default=False, help='')
    parser.add_argument('--find_unused_parameters', action='store_true', default=False, help='')
    parser.add_argument('--no_fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--oss_data', action='store_true', default=False, help='')

    parser.add_argument('--occupy', action='store_true', default=False, help='')
    
    parser.add_argument('--clean_shm', action='store_true', default=False, help='clean shm memory')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    # replace some default args
    if cfg.get('OTHERS', None):
        for key in cfg.OTHERS:
            setattr(args, key.lower(), cfg.OTHERS[key])

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.test_batch_size = cfg.OPTIMIZATION.get('TEST_BATCH_SIZE_PER_GPU', args.batch_size)

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs
    if args.occupy:
        args.epochs = 100000

    if not args.no_fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # use oss for data loading
    if args.oss_data or (cfg.get('OSS', None) and cfg.OSS.DATA):
        common_utils.oss_data_client = common_utils.OSSClient()
        logger.info(f'Ceph client initialization with root path at {cfg.DATA_CONFIG.OSS_PATH}')

    # For caption
    # if cfg.get('CAPTION', None) and cfg.CAPTION.ENABLED:
    #     caption_items = caption_utils.get_caption_items(cfg.CAPTION)
    # else:
    #     caption_items = None

    if args.clean_shm and cfg.LOCAL_RANK == 0:
        logger.info(">>> clean shm memory....")
        os.system("rm /dev/shm/scannet_* -f")
        os.system("rm /dev/shm/Area_* -f")
        logger.info(">>> clean shm memory done")
        
    commu_utils.synchronize()
    
    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None,
        multi_epoch_loader=args.multi_epoch_loader
    )

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.test_batch_size,
        dist=dist_train, workers=args.workers, logger=logger, training=False
    )

    if cfg.get('OFFSET_ST'):
        # different batch size, repeat != 1, x4_split cause we cannot use original train set here.
        pseudo_train_set, pseudo_train_loader, _ = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.test_batch_size,
            dist=dist_train, workers=args.workers, logger=logger,
            training=False, split='train'
        )
    else:
        pseudo_train_loader = None

    if cfg.get('TEXT_ENCODER', None) or cfg.MODEL.TASK_HEAD.get('TEXT_EMBED', None):
        text_encoder = build_text_network(cfg.TEXT_ENCODER).cuda()
        if cfg.get('TEXT_ENCODER', None) and cfg.TEXT_ENCODER.EXTRACT_EMBED:
            text_embed = load_text_embedding_from_encoder(cfg.TEXT_ENCODER, text_encoder, logger)
        else:
            text_embed = load_text_embedding_from_path(cfg.MODEL.TASK_HEAD.TEXT_EMBED, logger)
        cfg.MODEL.TASK_HEAD.TEXT_EMBED.CHANNEL = cfg.MODEL.ADAPTER.TEXT_DIM = text_embed.shape[1]
        cfg.MODEL.TASK_HEAD.TEXT_EMBED.NUM_CLASS = text_embed.shape[0]
    else:
        text_embed = None
        text_encoder = None

    model = build_vision_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn and total_gpus > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    commu_utils.synchronize()

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)

    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / 'last_train.pth'))
        if len(ckpt_list) == 0:
            ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    best_metric, best_epoch = load_best_metric(ckpt_dir)
    logger.info("=> loaded best metric '{}' (epoch {})".format(best_metric, best_epoch))

    if text_embed is not None:
        model.task_head.set_cls_head_with_text_embed(text_embed)

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()],
            find_unused_parameters=args.find_unused_parameters)
    logger.info(model)

    if args.merge_all_iters_to_one_epoch:
        dataset_length = int(len(train_loader) / args.epochs)
    else:
        dataset_length = len(train_loader)
    lr_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=dataset_length, total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    task = 'inst' if 'inst_class_idx' in cfg.DATA_CONFIG else 'sem'
    train_model(
        args,
        model,
        optimizer,
        train_loader,
        test_loader,
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        task=task,
        train_sampler=train_sampler,
        best_metric=best_metric,
        best_epoch=best_epoch,
        logger=logger,
        dist_train=dist_train,
        # caption_items=caption_items,
        text_encoder=text_encoder,
        pseudo_train_loader=pseudo_train_loader
    )


if __name__ == '__main__':
    main()

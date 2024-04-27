from .. import _init_path
import torch
import argparse
import tqdm
from pathlib import Path

from pcseg.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcseg.datasets import build_dataloader
from pcseg.utils import common_utils


def main(args, cfg):
    # TODO: enable dist test 
    dist_test = False
    logger = common_utils.create_logger(rank=cfg.LOCAL_RANK)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(test_loader), leave=True, desc='eval', dynamic_ncols=True)
    
    image_list = cfg.DATA_CONFIG.IMAGE_LIST
    for batch_dict in test_loader:
        # TODO: can only support batch size 1 evaluation now
        assert batch_dict['cam'].shape[0] == 1
        points = batch_dict['points'][:, 1:]
        point_img_feat_list = []
        for image_name in image_list:
            image_name = image_name.lower()
            image = torch.from_numpy(batch_dict['cam'][image_name]).cuda()
            image = image * 255
            
            features = extract_image_features(image)
            # x, y
            point_img = torch.from_numpy(batch_dict['point_img'][image_name]).cuda()
            
            # get feats from point_img
            point_img_feats = features[point_img[1], point_img[0]]
            point_img_idx = torch.from_numpy(batch_dict['point_img_idx'][image_name]).cuda()

            # TODO:
            

def extract_image_features(image):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
        
    main(args, cfg)

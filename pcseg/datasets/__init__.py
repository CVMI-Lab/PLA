import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler as _DistributedSampler

from pcseg.utils import common_utils

from .dataset import DatasetTemplate
from .scannet.scannet_dataset import ScanNetDataset, ScanNetInstDataset
from .s3dis.s3dis_dataset import S3DISDataset, S3DISInstDataset
from .nuscenes.nuscenes_dataset import NuScenesDataset, NuScenesPanopticDataset
from .stpls3d.stpls3d_dataset import STPLS3DDataset, STPLS3DInstDataset
from .kitti.kitti_dataset import KittiDataset, KittiPanopticDataset


__all__ = {
    'DatasetTemplate': DatasetTemplate,
    'ScanNetDataset': ScanNetDataset,
    'ScanNetInstDataset': ScanNetInstDataset,
    'S3DISDataset': S3DISDataset,
    'S3DISInstDataset': S3DISInstDataset,
    'NuScenesDataset': NuScenesDataset,
    'NuScenesPanopticDataset': NuScenesPanopticDataset,
    'KittiDataset': KittiDataset,
    'KittiPanopticDataset': KittiPanopticDataset,
    'STPLS3DDataset': STPLS3DDataset,
    'STPLS3DInstDataset': STPLS3DInstDataset
}


class DistributedSampler(_DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


def build_dataloader(dataset_cfg, class_names, batch_size, dist, root_path=None, workers=4, seed=None,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0,
                     multi_epoch_loader=False, split=None):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=root_path,
        training=training,
        logger=logger,
        split=split
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    if multi_epoch_loader:
        loader = MultiEpochsDataLoader
    else:
        loader = DataLoader

    dataloader = loader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, drop_last=False, sampler=sampler,
        collate_fn=getattr(dataset, dataset_cfg.COLLATE_FN),
        timeout=0, worker_init_fn=partial(common_utils.worker_init_fn, seed=seed)
    )

    return dataset, dataloader, sampler


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

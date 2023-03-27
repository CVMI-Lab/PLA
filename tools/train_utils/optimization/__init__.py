from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
import numpy as np

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


def build_optimizer(model, optim_cfg):
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    total_steps = total_iters_each_epoch * total_epochs

    if optim_cfg.SCHEDULER == 'poly':
        lr_scheduler = PolyLR(optimizer, max_iter=total_steps, power=optim_cfg.POWER)
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    else:
        lr_scheduler = None

    return lr_scheduler


class LambdaStepLR(lr_sched.LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_step=-1):
        super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

    @property
    def last_step(self):
        """Use last_epoch for the step counter"""
        return self.last_epoch

    @last_step.setter
    def last_step(self, v):
        self.last_epoch = v


class PolyLR(LambdaStepLR):
    """DeepLab learning rate policy"""
    def __init__(self, optimizer, max_iter, power=0.9, last_step=-1):
        super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_step)


class CosLR(LambdaStepLR):
    """Runyu's LR policy"""
    def __init__(self, optimizer, cos_lambda_func, last_step=-1):
        super(CosLR, self).__init__(optimizer, cos_lambda_func, last_step)


def cosine_lr_after_step(optimizer, base_lr, epoch, step_epoch, total_epochs, clip=1e-6):
    if epoch < step_epoch:
        lr = base_lr
    else:
        lr = clip + 0.5 * (base_lr - clip) * \
            (1 + np.cos(np.pi * ((epoch - step_epoch) / (total_epochs - step_epoch))))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lr(optim_cfg, optimizer, scheduler, total_epochs, total_iters_per_epoch, epoch, iter, accumulated_iter, no_step=False):
    # adjust learning rate
    if optim_cfg.SCHEDULER == 'cos':
        max_iter = total_iters_per_epoch * total_epochs
        cos_learning_rate(
            optimizer, optim_cfg.LR, epoch * total_iters_per_epoch + iter + 1, max_iter, 0, 0)
    elif optim_cfg.SCHEDULER == 'cos_after_step':
        cosine_lr_after_step(optimizer, optim_cfg.LR, epoch, optim_cfg.STEP_EPOCH, total_epochs)
    elif optim_cfg.SCHEDULER in ['adam_onecycle', 'poly']:
        assert scheduler is not None
        if not no_step:
            scheduler.step(accumulated_iter)
    elif optim_cfg.SCHEDULER in ['multistep']:
        pass
    else:
        raise NotImplementedError

import torch
import torch.nn as nn

from pcseg.config import cfg


class LinearHead(nn.Module):
    def __init__(self, model_cfg, in_channel, ignore_label, num_class):
        super(LinearHead, self).__init__()
        self.model_cfg = model_cfg
        self.in_channel = in_channel
        self.ignore_label = ignore_label
        self.num_class = num_class

        self.cls_head = nn.Linear(self.in_channel, self.num_class)

        self.valid_class_idx = [i for i in range(self.num_class)]
        if hasattr(cfg.DATA_CONFIG, 'ignore_class_idx'):
            self.ignore_class_idx = cfg.DATA_CONFIG.ignore_class_idx
            for i in self.ignore_class_idx:
                self.valid_class_idx.remove(i)

        self.seg_loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_label).cuda()
        self.forward_ret_dict = {}

    def forward(self, batch_dict):
        backbone3d_feats = batch_dict['backbone_3d_feats']

        semantic_scores = self.cls_head(backbone3d_feats)
        if self.training and self.model_cfg.get('VOXEL_LOSS', None):
            pass
        else:
            semantic_scores = semantic_scores[batch_dict['v2p_map']]

        semantic_scores = semantic_scores[..., self.valid_class_idx]
        semantic_preds = semantic_scores.max(1)[1]

        self.forward_ret_dict['seg_scores'] = semantic_scores
        self.forward_ret_dict['seg_preds'] = semantic_preds

        # save gt label to forward_ret_dict
        self.forward_ret_dict['seg_labels'] = batch_dict['labels']

    def get_loss(self):
        semantic_scores = self.forward_ret_dict['seg_scores']
        semantic_labels = self.forward_ret_dict['seg_labels']

        seg_loss = self.seg_loss_func(semantic_scores, semantic_labels)

        tb_dict = {'loss_seg': seg_loss.item()}
        return seg_loss, tb_dict

import os
import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict

from ..text_networks import load_text_embedding_from_path
from pcseg.utils import category_remap_utils
from pcseg.config import cfg
from pcseg.utils import common_utils


class TextSegHead(nn.Module):
    def __init__(self, model_cfg, in_channel, ignore_label, **kwargs):
        super(TextSegHead, self).__init__()
        self.model_cfg = model_cfg
        self.in_channel = in_channel
        self.ignore_label = ignore_label
        self.correct_seg_pred_binary = model_cfg.get('CORRECT_SEG_PRED_BINARY', True)
        self.eval_only = model_cfg.get('EVAL_ONLY', None)
        self.no_v2p_map = model_cfg.get('NO_V2P_MAP', False)
        self.text_channel = self.model_cfg.TEXT_EMBED.CHANNEL
        self.num_class = self.model_cfg.TEXT_EMBED.NUM_CLASS
        self.feat_norm = model_cfg.get('FEAT_NORM', False)

        # create cls head
        self.cls_head = nn.Linear(self.text_channel, self.num_class, bias=False)

        if model_cfg.get('LOGIT_SCALE', None):
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) if model_cfg.LOGIT_SCALE.learnable else model_cfg.LOGIT_SCALE.value
        else:
            self.logit_scale = 1.0

        # fix text cls head
        for param in self.cls_head.parameters():
            param.requires_grad = False

        # open vocab
        self.valid_class_idx = [i for i in range(len(cfg.CLASS_NAMES))]
        if hasattr(cfg.DATA_CONFIG, 'base_class_idx'):
            self.base_class_idx = cfg.DATA_CONFIG.base_class_idx
            self.novel_class_idx = cfg.DATA_CONFIG.novel_class_idx
        if hasattr(cfg.DATA_CONFIG, 'ignore_class_idx'):
            self.ignore_class_idx = cfg.DATA_CONFIG.ignore_class_idx
            for i in self.ignore_class_idx:
                self.valid_class_idx.remove(i)

        # remap category name for ambigous categories
        self.need_class_mapping = self.model_cfg.get('CLASS_MAPPING', False)
        if self.need_class_mapping:
            self.idx_mapping = category_remap_utils.cast_category_name_mapping_to_idx_mapping(
                self.model_cfg.CLASS_MAPPING, cfg.TEXT_ENCODER.CATEGORY_NAMES, cfg.CLASS_NAMES
            )

        self.seg_loss_weight = model_cfg.get('SEMANTIC_WEIGHT', None)
        if self.seg_loss_weight is not None:
            self.seg_loss_weight = torch.FloatTensor(self.seg_loss_weight).cuda()
            if hasattr(cfg.DATA_CONFIG, 'base_class_idx'):
                self.seg_loss_weight = self.seg_loss_weight[self.base_class_idx]
            else:
                self.seg_loss_weight = self.seg_loss_weight[self.valid_class_idx]
        self.seg_loss_func = nn.CrossEntropyLoss(
            weight=self.seg_loss_weight, ignore_index=self.ignore_label).cuda()
        self.forward_ret_dict = {}

    def set_cls_head_with_text_embed(self, text_embed):
        self.cls_head.load_state_dict(OrderedDict({'weight': text_embed.float()}))

    def forward(self, batch_dict):
        self.forward_ret_dict = {}
        # if self.eval_only and self.training:
        #     return batch_dict

        adapter_feats = batch_dict['adapter_feats']
        if self.feat_norm:
            adapter_feats = nn.functional.normalize(adapter_feats, dim=-1)

        if isinstance(self.logit_scale, nn.Parameter):
            logit_scale = self.logit_scale.exp()
        else:
            logit_scale = self.logit_scale

        semantic_scores = self.cls_head(adapter_feats) * logit_scale

        if self.no_v2p_map or (self.training and self.model_cfg.get('VOXEL_LOSS', None)):
            pass
        else:
            semantic_scores = semantic_scores[batch_dict['v2p_map']]
        if not self.training and batch_dict['test_x4_split']:
            semantic_scores = common_utils.merge_4_parts(semantic_scores)

        if not self.training and self.need_class_mapping:
            semantic_scores = self.remap_category(semantic_scores, self.idx_mapping)

        if self.training or batch_dict.get('pseudo_label_generation', False):
            if hasattr(self, 'base_class_idx'):
                semantic_scores = semantic_scores[..., self.base_class_idx]
            else:
                semantic_scores = semantic_scores[..., self.valid_class_idx]
        else:
            new_semantic_scores = semantic_scores.detach().clone()
            new_semantic_scores[:] = -1e6
            new_semantic_scores[..., self.valid_class_idx] = semantic_scores[..., self.valid_class_idx]
            semantic_scores = new_semantic_scores

        # get semantic prediction results
        # consider the binary calibrate 
        if (not self.training) and (not batch_dict.get('pseudo_label_generation', False)) and \
            batch_dict.get('binary_ret_dict') and self.correct_seg_pred_binary:
            binary_preds, semantic_preds = self.correct_seg_pred_with_binary_pred(
                batch_dict, semantic_scores
            )
        else:
            semantic_preds = semantic_scores.max(1)[1]
            if batch_dict.get('binary_ret_dict', None):
                binary_preds = batch_dict['binary_ret_dict']['binary_preds']
            else:
                binary_preds = None

        # for 2D fusion
        if not self.training and not batch_dict.get('pseudo_label_generation', False) and \
            'adapter_feats_mask' in batch_dict:
            semantic_preds[~batch_dict['adapter_feats_mask'].bool().cuda()] = self.ignore_label

        # for captions
        batch_dict['seg_scores'] = semantic_scores
        batch_dict['seg_preds'] = semantic_preds

        self.forward_ret_dict['seg_scores'] = semantic_scores
        self.forward_ret_dict['seg_preds'] = semantic_preds
        self.forward_ret_dict['binary_preds'] = binary_preds

        # save gt label to forward_ret_dict
        self.forward_ret_dict['seg_labels'] = batch_dict['labels']

        return batch_dict

    def get_loss(self):
        semantic_scores = self.forward_ret_dict['seg_scores']
        semantic_labels = self.forward_ret_dict['seg_labels']
        seg_loss = self.seg_loss_func(semantic_scores, semantic_labels) * self.model_cfg.get('LOSS_WEIGHT', 1.0)

        tb_dict = {'loss_seg': seg_loss.item()}
        return seg_loss, tb_dict

    def correct_seg_pred_with_binary_pred(self, batch_dict, semantic_scores):
        binary_preds = batch_dict['binary_ret_dict']['binary_preds']
        binary_scores = batch_dict['binary_ret_dict']['binary_scores']

        base_semantic_scores = semantic_scores[..., self.base_class_idx].softmax(dim=-1)
        novel_semantic_scores = semantic_scores[..., self.novel_class_idx].softmax(dim=-1)
        semantic_scores = semantic_scores.clone()
        semantic_scores[:] = 0.
        semantic_scores[..., self.base_class_idx] = base_semantic_scores
        semantic_scores[..., self.novel_class_idx] = novel_semantic_scores
        sigmoid_binary_scores = torch.sigmoid(binary_scores) 
        sigmoid_binary_scores = sigmoid_binary_scores.repeat(1, semantic_scores.shape[-1])
        sigmoid_binary_scores[..., self.novel_class_idx] = 1 - sigmoid_binary_scores[..., self.novel_class_idx]

        semantic_scores = semantic_scores * sigmoid_binary_scores
        semantic_scores /= semantic_scores.sum(-1, keepdim=True)
        semantic_preds = semantic_scores.max(1)[1]
        return binary_preds, semantic_preds

    @staticmethod
    def remap_category(semantic_scores, idx_mapping):
        new_semantic_scores = torch.zeros((semantic_scores.shape[0], len(cfg.CLASS_NAMES)), dtype=torch.float32).cuda()
        for idx in range(new_semantic_scores.shape[1]):
            if isinstance(idx_mapping[idx], list):
                source_idxs = torch.from_numpy(np.array(idx_mapping[idx], dtype=np.int64)).cuda()
                selected_logits = semantic_scores[..., source_idxs]
                selected_idx = selected_logits.max(1)[1]
                max_score_idx = source_idxs[selected_idx]
                new_semantic_scores[..., idx] = semantic_scores[
                    torch.arange(semantic_scores.shape[0]).cuda(), max_score_idx
                ]
            else:
                new_semantic_scores[..., idx] = semantic_scores[..., idx_mapping[idx]]

        return new_semantic_scores

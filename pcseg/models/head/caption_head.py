import torch
import torch.nn as nn
import numpy as np
from ..model_utils.fp16 import force_fp32
from pcseg.utils import loss_utils
from torch_scatter import scatter_mean


class CaptionHead(nn.Module):
    def __init__(self, model_cfg, ignore_label):
        super().__init__()
        self.model_cfg = model_cfg
        self.feat_norm = model_cfg.FEAT_NORM
        self.in_feat_name = model_cfg.get('IN_FEAT_NAME', 'adapter_feats')

        if model_cfg.LOGIT_SCALE.learnable:
            self.caption_logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.caption_logit_scale = model_cfg.LOGIT_SCALE.value

        self.caption_loss_func = self.build_loss_func(ignore_label)
        self.caption_loss_weight = model_cfg.LOSS_WEIGHT

        self.forward_ret_dict = {}

    def build_loss_func(self, ignore_label):
        loss_type = self.model_cfg.get('LOSS_FUNC', 'CrossEntropy')
        if loss_type == 'CrossEntropy':  # pooling features
            caption_loss_func = nn.CrossEntropyLoss(ignore_index=ignore_label).cuda()
        elif loss_type == 'BYOL':
            caption_loss_func = loss_utils.BYOLLoss()
        else:
            raise NotImplementedError

        return caption_loss_func

    def forward(self, batch_dict):
        self.forward_ret_dict = {}
        if not self.training:
            return batch_dict

        caption_infos = batch_dict['caption_infos']
        v2p_map = batch_dict['v2p_map']
        adapter_feats = batch_dict['adapter_feats'][v2p_map]

        if isinstance(self.caption_logit_scale, nn.Parameter):
            caption_logit_scale = self.caption_logit_scale.exp()
        else:
            caption_logit_scale = self.caption_logit_scale

        caption_ret_dict = {}
        # caption_types = ['caption_scene', 'caption_view', 'caption_entity']
        for caption_type in caption_infos.keys():

            # pooling object
            caption_embed = caption_infos[caption_type]['caption_embed']
            if caption_embed.shape[0] == 0:
                caption_ret_dict[caption_type] = {'zero_loss': adapter_feats.sum() * 0.}
                continue

            # forward pooling function
            if caption_type == 'caption_scene':
                caption_ret_dict[caption_type] = self.forward_scene_caption(
                    batch_dict, caption_infos[caption_type], adapter_feats, caption_logit_scale
                )
            else:
                caption_ret_dict[caption_type] = self.forward_given_type_caption(
                    batch_dict, caption_infos[caption_type], adapter_feats, caption_logit_scale
                )
        self.forward_ret_dict = caption_ret_dict
        return batch_dict

    @force_fp32(apply_to=('adapter_feats'))
    def forward_scene_caption(self, batch_dict, caption_info, adapter_feats, logit_scale):
        caption_embed = caption_info['caption_embed']
        caption_idx = caption_info['caption_idx']
        batch_idx = batch_dict['batch_idxs']

        pooled_feats = scatter_mean(adapter_feats, index=batch_idx.long(), dim=0)
        exist_caption_idx = torch.ones(pooled_feats.shape[0]).bool().cuda()
        for ii in range(exist_caption_idx.shape[0]):
            if len(caption_info['select_image_corr'][ii]) == 0:
                exist_caption_idx[ii] = False
        pooled_feats = pooled_feats[exist_caption_idx]

        caption_logit, caption_labels = self.prepare_caption_loss_logit_and_labels(
            pooled_feats, caption_embed, logit_scale, caption_idx
        )
        ret_dict = {'caption_output': caption_logit, 'caption_labels': caption_labels}
        return ret_dict

    def forward_given_type_caption(self, batch_dict, caption_info, adapter_feats, logit_scale):
        ret_dict = {}
        caption_func = self._forward_given_type_caption
        pooled_feats, real_n_points, if_has_pts = caption_func(
            batch_dict, caption_info, adapter_feats
        )
        # if some scene don't have suitable image, len(pooled_features) == 0
        if len(pooled_feats) > 0:
            pooled_feats = torch.cat(pooled_feats, 0)
            real_n_points = torch.cat(real_n_points, 0)
            exist_caption_idx = torch.cat(if_has_pts, 0)

            caption_idx = caption_info['caption_idx']
            normed_caption_embed = caption_info['caption_embed']

            caption_logit, caption_labels = self.prepare_caption_loss_logit_and_labels(
                pooled_feats, normed_caption_embed, logit_scale, caption_idx, exist_caption_idx
            )
            ret_dict = {'caption_output': caption_logit, 'caption_labels': caption_labels,
                        'caption_n_points': real_n_points}

        ret_dict['zero_loss'] = 0.0 * adapter_feats.sum() * logit_scale
        return ret_dict

    def _forward_given_type_caption(self, batch_dict, caption_info, adapter_feats):
        frame_corr_idx = caption_info['select_image_corr']

        pooled_feats = []
        real_n_points = []
        if_has_pts = []
        batch_idx = batch_dict['batch_idxs']

        for b in range(len(frame_corr_idx)):
            _frame_corr_idx = frame_corr_idx[b]
            offsets = batch_dict['offsets']
            origin_idx = batch_dict['origin_idx'][offsets[b]: offsets[b + 1]] if 'origin_idx' in batch_dict else None
            pc_count = batch_dict['pc_count'][b]
            batch_if_has_pts = torch.zeros(len(_frame_corr_idx), dtype=torch.bool).cuda()
            for i, idx in enumerate(_frame_corr_idx):
                selected_mask = self.get_point_mask_for_point_img_points(pc_count, idx, origin_idx)

                # visualization debug code
                # import tools.visual_utils.open3d_vis_utils as vis
                # points = batch_dict['points'][batch_idx == b]
                # points_batch = points[selected_mask]
                # points_colors = batch_dict['rgb'][batch_idx == b][selected_mask]
                #
                # vis_dict = {
                #     'points': points_batch[:, 1:].detach().cpu().numpy(),
                #     'point_colors': points_colors.detach().cpu().numpy(),
                #     'point_size': 2.0
                # }
                # vis.dump_vis_dict(vis_dict, './vis_dict_2.pkl')
                # import ipdb; ipdb.set_trace(context=20)

                _pooled_feats = adapter_feats[batch_idx == b][selected_mask]
                batch_if_has_pts[i] = selected_mask.sum() > 0
                if selected_mask.sum() > 0:
                    real_n_points.append(selected_mask.sum().view(1))
                    pooled_feats.append(_pooled_feats.mean(0, keepdim=True))
            if_has_pts.append(batch_if_has_pts)

        return pooled_feats, real_n_points, if_has_pts

    def prepare_caption_loss_logit_and_labels(self, pooled_features, caption_embed,
                                              caption_logit_scale, caption_idx, exist_caption_idx=None):
        if self.feat_norm:
            pooled_features = nn.functional.normalize(pooled_features, dim=-1)
        normed_caption_embed = nn.functional.normalize(caption_embed, dim=-1)

        loss_type = self.model_cfg.get('LOSS_FUNC', 'CrossEntropy')
        if loss_type == 'CrossEntropy':
            caption_logit = pooled_features @ normed_caption_embed.float().T * caption_logit_scale

            if exist_caption_idx is not None:
                caption_labels = caption_idx[exist_caption_idx]
            else:
                caption_labels = caption_idx
        elif loss_type == 'BYOL':
            caption_logit = pooled_features
            caption_labels = normed_caption_embed
            if exist_caption_idx is not None:
                caption_labels = caption_labels[exist_caption_idx]
        else:
            raise NotImplementedError

        return caption_logit, caption_labels

    @staticmethod
    def get_point_mask_for_point_img_points(pc_count, idx, origin_idx):
        """

        Args:
            pc_count:
            idx:
            origin_idx:

        Returns:

        """
        selected_mask = torch.zeros(pc_count, dtype=torch.bool)
        if idx is None:  # scene caption don't need to consider this part
            selected_mask[:] = True
        else:
            selected_mask[idx.long()] = True
        if origin_idx is not None:
            selected_mask = selected_mask[origin_idx]

        selected_mask = selected_mask.cuda(non_blocking=True)

        return selected_mask

    def get_loss(self):
        caption_loss = 0
        tb_dict = {}
        for caption_type in self.forward_ret_dict:
            if 'caption_output' in self.forward_ret_dict[caption_type]:
                caption_output = self.forward_ret_dict[caption_type]['caption_output']
                caption_labels = self.forward_ret_dict[caption_type]['caption_labels']
                cur_caption_loss_weight = self.caption_loss_weight[caption_type.split('_')[-1].upper()]
                cur_caption_loss = self.caption_loss_func(caption_output,  caption_labels) * cur_caption_loss_weight
                tb_dict[caption_type] = cur_caption_loss.item()
            else:
                tb_dict[caption_type] = 0.0
                # if some GPUs don't have loss, some GPUs have loss for backward, the process will stuck
                cur_caption_loss = self.forward_ret_dict[caption_type]['zero_loss']

            caption_loss += cur_caption_loss

        return caption_loss, tb_dict

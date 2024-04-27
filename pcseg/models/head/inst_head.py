import os
import torch
import functools
import torch.nn as nn
import numpy as np
from torch_scatter import scatter_mean
from sklearn.cluster import AgglomerativeClustering

from ...external_libs.softgroup_ops.ops import functions as ops
from ..model_utils.unet_blocks import UBlock, ResidualBlock, VGGBlock
from ..model_utils.basic_block_1d import MLP
from ..model_utils.rle_utils import rle_encode, rle_decode
from ..model_utils.fp16 import force_fp32
from pcseg.config import cfg
from pcseg.utils import common_utils, loss_utils
from ...utils.spconv_utils import spconv


class InstHead(nn.Module):
    def __init__(self, model_cfg, in_channel, inst_class_idx, sem2ins_classes,
                 valid_class_idx, label_shift, ignore_label=-100,
                 base_inst_class_idx=None, novel_inst_class_idx=None):
        super().__init__()

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        self.model_cfg = model_cfg
        # self.in_channel = in_channel
        self.mid_channel = in_channel
        self.block_residual = model_cfg.BLOCK_RESIDUAL
        self.semantic_only = model_cfg.SEMANTIC_ONLY
        self.inst_class_idx = np.array(inst_class_idx)
        # self.semantic_weight = semantic_weight
        self.sem2ins_classes = sem2ins_classes
        self.valid_class_idx = valid_class_idx
        self.label_shift = label_shift
        self.ignore_label = ignore_label
        # self.with_coords = with_coords
        self.prepare_epoch = model_cfg.CLUSTERING.PREPARE_EPOCH
        self.grouping_cfg = model_cfg.CLUSTERING.GROUPING_CFG
        self.inst_voxel_cfg = model_cfg.CLUSTERING.INST_VOXEL_CFG
        keys = list(self.inst_voxel_cfg.keys())
        for key in keys:
            self.inst_voxel_cfg[key.lower()] = self.inst_voxel_cfg[key]
            del self.inst_voxel_cfg[key]
        self.loss_cfg = model_cfg.CLUSTERING.LOSS_CFG
        self.test_cfg = model_cfg.CLUSTERING.TEST_CFG
        self.fixed_modules = model_cfg.FIXED_MODULES
        self.in_feature_name = model_cfg.get('IN_FEAT_NAME', 'backbone_3d_feats')
        self.no_v2p_map = model_cfg.get('NO_V2P_MAP', False)

        # point-wise prediction
        self.offset_linear = MLP([self.mid_channel, self.mid_channel, 3], norm_fn=norm_fn, num_layers=2)
        self.offset_loss_type = model_cfg.get('OFFSET_LOSS', 'l1')
        # top-down refinement
        if self.block_residual:
            block = functools.partial(ResidualBlock, custom_sp1x1=self.model_cfg.get('CUSTOM_SP1X1', False))
        else:
            block = VGGBlock
        self.tiny_unet = UBlock([self.mid_channel, 2 * self.mid_channel], norm_fn, 2, block, indice_key_id=11)
        self.tiny_unet_outputlayer = spconv.SparseSequential(norm_fn(self.mid_channel), nn.ReLU())
        # self.cls_linear = nn.Linear(self.mid_channel, self.num_inst_classes + 1)
        self.mask_linear = MLP([self.mid_channel, self.mid_channel, 1], norm_fn=None, num_layers=2)
        self.iou_score_linear = nn.Linear(self.mid_channel, 1)

        # init parameters
        self.apply(self.set_bn_init)

        for mod in self.fixed_modules:
            mod = getattr(self, mod)
            for param in mod.parameters():
                param.requires_grad = False

        # open vocab
        if hasattr(cfg.DATA_CONFIG, 'base_class_idx'):
            self.base_class_idx = cfg.DATA_CONFIG.base_class_idx
            self.novel_class_idx = cfg.DATA_CONFIG.novel_class_idx
            assert base_inst_class_idx is not None
            self.inst_base_class_idx = base_inst_class_idx
            self.inst_novel_class_idx = novel_inst_class_idx
            self.train_sem_classes = cfg.DATA_CONFIG.base_class_idx
        else:
            self.train_sem_classes = self.valid_class_idx
        self.test_sem_classes = self.valid_class_idx
        self.correct_seg_pred_binary = model_cfg.get('CORRECT_SEG_PRED_BINARY', True)

        # offset pseudo generation
        if model_cfg.CLUSTERING.get('PSEUDO_OFFSET_CFG', False):
            self.pseudo_offset_cfg = model_cfg.CLUSTERING.PSEUDO_OFFSET_CFG

        self.forward_ret_dict = {}

    @staticmethod
    def set_bn_init(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.0)
            m.bias.data.fill_(0.0)

    def get_batch_offsets(self, batch_idxs, bs):
        batch_offsets = torch.zeros(bs + 1).int().cuda()
        for i in range(bs):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets

    @force_fp32(apply_to=('x'))
    def global_pool(self, x, expand=False):
        indices = x.indices[:, 0]
        batch_counts = torch.bincount(indices)
        batch_offset = torch.cumsum(batch_counts, dim=0)
        pad = batch_offset.new_full((1, ), 0)
        batch_offset = torch.cat([pad, batch_offset]).int()
        x_pool = ops.global_avg_pool(x.features, batch_offset)
        if not expand:
            return x_pool

        x_pool_expand = x_pool[indices.long()]
        x.features = torch.cat((x.features, x_pool_expand), dim=1)
        return x

    @force_fp32(apply_to=('semantic_scores, pt_offsets'))
    def forward_grouping(self, batch_size, semantic_scores, pt_offsets, batch_idxs, coords_float,
                         super_voxel=None, output_feats=None, binary_scores=None, binary_labels=None, 
                         pseudo_label_generation=False, point_nth_multiplier=1.0, **kwargs):
        proposals_idx_list = []
        proposals_offset_list = []
        proposals_sem_scores = []
        binary_scores_list = []

        _semantic_scores = semantic_scores.clone()
        if not self.training and (not pseudo_label_generation) and self.correct_seg_pred_binary and \
            binary_scores is not None:
            base_semantic_scores = semantic_scores[..., self.base_class_idx].softmax(dim=-1)
            novel_semantic_scores = semantic_scores[..., self.novel_class_idx].softmax(dim=-1)
            semantic_scores = semantic_scores.clone()
            semantic_scores[:] = 0
            semantic_scores[..., self.base_class_idx] = base_semantic_scores
            semantic_scores[..., self.novel_class_idx] = novel_semantic_scores
            sigmoid_binary_scores = torch.sigmoid(binary_scores / 1.0)  # novel: 0, base: 1
            sigmoid_binary_scores = sigmoid_binary_scores.repeat(1, semantic_scores.shape[1])
            sigmoid_binary_scores[..., self.novel_class_idx] = 1 - sigmoid_binary_scores[..., self.novel_class_idx]
            semantic_scores = semantic_scores * sigmoid_binary_scores
            semantic_scores /= semantic_scores.sum(-1, keepdim=True)
        else:
            semantic_scores = semantic_scores.softmax(dim=-1)

        radius = self.grouping_cfg.RADIUS
        mean_active = self.grouping_cfg.MEAN_ACTIVE
        npoint_thr = int(self.grouping_cfg.NPOINT_THR * point_nth_multiplier)
        class_numpoint_mean = torch.tensor(
            self.grouping_cfg.CLASS_NUMPOINT_MEAN, dtype=torch.float32)

        if self.training or pseudo_label_generation:
            class_id_curr = np.arange(len(self.train_sem_classes))
            class_id_origin = self.train_sem_classes
        else: 
            class_id_curr = class_id_origin = self.test_sem_classes

        for cic, cio in zip(class_id_curr, class_id_origin):  # only training
            if cio in self.grouping_cfg.IGNORE_CLASSES:
                continue
            scores = semantic_scores[:, cic].contiguous()
            object_idxs = (scores > self.grouping_cfg.SCORE_THR).nonzero().view(-1)
            if object_idxs.size(0) < self.test_cfg.MIN_NPOINT:
                continue
            batch_idxs_ = batch_idxs[object_idxs]
            batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
            coords_ = coords_float[object_idxs]
            pt_offsets_ = pt_offsets[object_idxs]
            idx, start_len = ops.ballquery_batch_p(
                coords_ + pt_offsets_, batch_idxs_, batch_offsets_, radius, mean_active)
            proposals_idx, proposals_offset = ops.bfs_cluster(
                class_numpoint_mean, idx.cpu(), start_len.cpu(), npoint_thr, cio)
            proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
            cls_scores = scatter_mean(_semantic_scores[proposals_idx[:, 1].long().cuda()], index=proposals_idx[..., 0].long().cuda(), dim=0)
            if (self.training or pseudo_label_generation) and hasattr(self, 'inst_base_class_idx'):
                proposals_sem_scores.append(cls_scores[..., self.label_shift: self.label_shift + len(self.inst_base_class_idx)])
            else:
                proposals_sem_scores.append(cls_scores[..., self.inst_class_idx])
            if binary_scores is not None:
                _binary_scores = scatter_mean(binary_scores[proposals_idx[:, 1].long().cuda()], index=proposals_idx[..., 0].long().cuda(), dim=0)
                binary_scores_list.append(_binary_scores)

            # merge proposals
            if len(proposals_offset_list) > 0:
                proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                proposals_offset += proposals_offset_list[-1][-1]
                proposals_offset = proposals_offset[1:]
            if proposals_idx.size(0) > 0:
                proposals_idx_list.append(proposals_idx)
                proposals_offset_list.append(proposals_offset)

        if pseudo_label_generation:
            for ii in range(0, 1):  # base: 1, novel: 0
                class_idx = (binary_labels == ii).nonzero().view(-1)
                if class_idx.shape[0] < self.test_cfg.MIN_NPOINT:
                    continue
                if super_voxel is not None:
                    _sv = super_voxel[class_idx]
                    _sv_set = torch.unique(_sv)
                    mapper = torch.zeros(_sv.max().item() + 1).long().cuda()
                    mapper[_sv_set] = torch.arange(len(_sv_set)).cuda()
                    _new_sv = mapper[_sv]
                    if len(_sv_set) < 2:
                        continue
                    sv_feats = scatter_mean(output_feats[class_idx], _new_sv, dim=0)
                    sv_clustering = AgglomerativeClustering(distance_threshold=self.pseudo_offset_cfg.CLUSTER_THR, n_clusters=None).fit(
                        sv_feats.detach().cpu().numpy())
                    sv_clustered_labels = torch.from_numpy(sv_clustering.labels_).cuda()
                    clustered_labels = sv_clustered_labels[_new_sv]
                    cluster_ids = torch.unique(sv_clustered_labels)
                else:
                    clustered_labels = torch.zeros(class_idx.shape[0]).long().cuda()
                    cluster_ids = [0]

                for jj in cluster_ids:
                    object_idxs = (clustered_labels == jj).nonzero().view(-1)
                    if object_idxs.size(0) < self.pseudo_offset_cfg.MIN_SV:
                        continue
                    batch_idxs_ = batch_idxs[class_idx][object_idxs]
                    batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
                    coords_ = coords_float[class_idx][object_idxs]
                    pt_offsets_ = pt_offsets[class_idx][object_idxs]
                    idx, start_len = ops.ballquery_batch_p(
                        coords_ + pt_offsets_, batch_idxs_, batch_offsets_, radius, mean_active)
                    proposals_idx, proposals_offset = ops.bfs_cluster(
                        class_numpoint_mean, idx.cpu(), start_len.cpu(), npoint_thr, 0)  # class_id set to zero
                    proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
                    proposals_idx[:, 1] = class_idx[proposals_idx[:, 1].long()].int()
                    cls_scores = scatter_mean(
                        _semantic_scores[proposals_idx[:, 1].long().cuda()].float(),
                        index=proposals_idx[..., 0].long().cuda(), dim=0)
                    proposals_sem_scores.append(cls_scores[..., self.label_shift: self.label_shift + len(self.inst_base_class_idx)])
                    # TODO: check
                    if binary_scores is not None:
                        _binary_scores = scatter_mean(
                            binary_scores[proposals_idx[:, 1].long().cuda()].float(),
                            index=proposals_idx[..., 0].long().cuda(), dim=0)
                        binary_scores_list.append(_binary_scores)

                    # merge proposals
                    if len(proposals_offset_list) > 0:
                        proposals_idx[:, 0] += sum([x.size(0) for x in proposals_offset_list]) - 1
                        proposals_offset += proposals_offset_list[-1][-1]
                        proposals_offset = proposals_offset[1:]
                    if proposals_idx.size(0) > 0:
                        proposals_idx_list.append(proposals_idx)
                        proposals_offset_list.append(proposals_offset)

        if len(proposals_idx_list) > 0:
            proposals_idx = torch.cat(proposals_idx_list, dim=0)
            proposals_offset = torch.cat(proposals_offset_list)
            proposals_sem_scores = torch.cat(proposals_sem_scores, dim=0)
            if binary_scores is not None:
                proposal_binary_scores = torch.cat(binary_scores_list, dim=0)
            else:
                proposal_binary_scores = None
            return proposals_idx, proposals_offset, proposals_sem_scores, proposal_binary_scores
        else:
            return proposals_idx_list, proposals_offset_list, proposals_sem_scores, binary_scores_list

    @force_fp32(apply_to=('feats'))
    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, scale, spatial_shape, 
                              rand_quantize=False):
        batch_idx = clusters_idx[:, 0].cuda().long()
        c_idxs = clusters_idx[:, 1].cuda()
        feats = feats[c_idxs.long()]
        coords = coords[c_idxs.long()]

        coords_min = ops.sec_min(coords, clusters_offset.cuda())
        coords_max = ops.sec_max(coords, clusters_offset.cuda())

        # 0.01 to ensure voxel_coords < spatial_shape
        clusters_scale = 1 / ((coords_max - coords_min) / spatial_shape).max(1)[0] - 0.01
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        coords_min = coords_min * clusters_scale[:, None]
        coords_max = coords_max * clusters_scale[:, None]
        clusters_scale = clusters_scale[batch_idx]
        coords = coords * clusters_scale[:, None]

        if rand_quantize:
            # after this, coords.long() will have some randomness
            range = coords_max - coords_min
            coords_min -= torch.clamp(spatial_shape - range - 0.001, min=0) * torch.rand(3).cuda()
            coords_min -= torch.clamp(spatial_shape - range + 0.001, max=0) * torch.rand(3).cuda()
        coords_min = coords_min[batch_idx]
        coords -= coords_min
        assert coords.shape.numel() == ((coords >= 0) * (coords < spatial_shape)).sum()
        coords = coords.long()
        coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), coords.cpu()], 1)

        out_coords, inp_map, out_map = ops.voxelization_idx(coords, int(clusters_idx[-1, 0]) + 1)
        out_feats = ops.voxelization(feats, out_map.cuda())
        spatial_shape = [spatial_shape] * 3
        voxelization_feats = spconv.SparseConvTensor(
            out_feats, out_coords.int().cuda(), spatial_shape, int(clusters_idx[-1, 0]) + 1
        )
        return voxelization_feats, inp_map

    def forward_instance(self, inst_feats, inst_map, **kwargs):
        feats = self.tiny_unet(inst_feats)
        feats = self.tiny_unet_outputlayer(feats)

        # predict mask scores
        mask_scores = self.mask_linear(feats.features)
        mask_scores = mask_scores[inst_map.long()]
        instance_batch_idxs = feats.indices[:, 0][inst_map.long()]

        # predict instance cls and iou scores
        feats = self.global_pool(feats)
        # cls_scores = self.semantic_linear(self.adapter(feats))
        # cls_scores = cls_scores[..., self.inst_class_idx]
        iou_scores = self.iou_score_linear(feats)

        return instance_batch_idxs, iou_scores, mask_scores

    @force_fp32(apply_to=('semantic_scores', 'cls_scores', 'iou_scores', 'mask_scores'))
    def get_instances(self, scan_id, proposals_idx, semantic_scores, cls_scores, iou_scores,
                      mask_scores, binary_scores=None, proposal_binary_scores=None):
        num_instances = cls_scores.size(0)
        num_points = semantic_scores.size(0)

        if self.correct_seg_pred_binary and binary_scores is not None:
            assert proposal_binary_scores is not None
            base_cls_scores = cls_scores[..., self.inst_base_class_idx].softmax(dim=-1)
            novel_cls_scores = cls_scores[..., self.inst_novel_class_idx].softmax(dim=-1)
            cls_scores = cls_scores.clone()
            cls_scores[:] = 0.
            cls_scores[..., self.inst_base_class_idx] = base_cls_scores
            cls_scores[..., self.inst_novel_class_idx] = novel_cls_scores

            base_semantic_scores = semantic_scores[..., self.inst_base_class_idx].softmax(dim=-1)
            novel_semantic_scores = semantic_scores[..., self.inst_novel_class_idx].softmax(dim=-1)
            semantic_scores = semantic_scores.clone()
            semantic_scores[:] = 0.
            semantic_scores[..., self.inst_base_class_idx] = base_semantic_scores
            semantic_scores[..., self.inst_novel_class_idx] = novel_semantic_scores

            sigmoid_binary_scores = torch.sigmoid(proposal_binary_scores / 1.0)  # novel: 0, base: 1
            sigmoid_binary_scores = sigmoid_binary_scores.repeat(1, cls_scores.shape[1])
            sigmoid_binary_scores[..., self.inst_novel_class_idx] = 1 - sigmoid_binary_scores[..., self.inst_novel_class_idx]
            cls_scores = cls_scores * sigmoid_binary_scores
            cls_scores /= cls_scores.sum(-1, keepdim=True)

            sigmoid_binary_scores = torch.sigmoid(binary_scores / 1.0)  # novel: 0, base: 1
            sigmoid_binary_scores = sigmoid_binary_scores.repeat(1, semantic_scores.shape[1])
            sigmoid_binary_scores[..., self.novel_class_idx] = 1 - sigmoid_binary_scores[..., self.novel_class_idx]
            semantic_scores = semantic_scores * sigmoid_binary_scores
            semantic_scores /= semantic_scores.sum(-1, keepdim=True)
        else:
            cls_scores = cls_scores.softmax(1)

        semantic_pred = semantic_scores.max(1)[1]
        cls_pred_list, score_pred_list, mask_pred_list = [], [], []
        # new_proposals_idx_list = []
        # old_proposal_idx = -1
        for i in (self.inst_class_idx - self.label_shift):  # TODO: check
            if i in self.sem2ins_classes:
                cls_pred = cls_scores.new_tensor([i + 1], dtype=torch.long)
                score_pred = cls_scores.new_tensor([1.], dtype=torch.float32)
                mask_pred = (semantic_pred == i)[None, :].int()
            else:
                cls_pred = cls_scores.new_full((num_instances, ), i + 1, dtype=torch.long)
                cur_cls_scores = cls_scores[:, i]
                cur_iou_scores = iou_scores[:, 0]  # [:, i]
                cur_mask_scores = mask_scores[:, 0]  # [:, i]
                score_pred = cur_cls_scores * torch.sigmoid(cur_iou_scores)  # .clamp(0, 1)

                mask_pred = torch.zeros((num_instances, num_points), dtype=torch.int8).cuda()
                mask_inds = cur_mask_scores > self.test_cfg.MASK_SCORE_THR
                cur_proposals_idx = proposals_idx[mask_inds.cpu()].long()
                mask_pred[cur_proposals_idx[:, 0], cur_proposals_idx[:, 1]] = 1

                # filter low score instance
                inds = cur_cls_scores > self.test_cfg.CLS_SCORE_THR
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]
                # new_proposals_idx = proposals_idx[inds[proposals_idx[..., 0].long()]]
                # remapper = torch.zeros(inds.shape[0] * self.instance_classes, dtype=torch.int)
                # remapper[torch.unique(new_proposals_idx[..., 0]).long()] = torch.arange(mask_pred.shape[0]).int()
                # new_proposals_idx[..., 0] = remapper[new_proposals_idx[..., 0].long()]

                # filter too small instances
                npoint = mask_pred.sum(1)
                inds = npoint >= self.test_cfg.MIN_NPOINT
                cls_pred = cls_pred[inds]
                score_pred = score_pred[inds]
                mask_pred = mask_pred[inds]
                # new_proposals_idx = new_proposals_idx[inds[new_proposals_idx[..., 0].long()]]
                # new_proposals_idx[..., 0] = new_proposals_idx[..., 0] + old_proposal_idx + 1
                # if new_proposals_idx.shape[0] > 0:
                #     old_proposal_idx = new_proposals_idx[..., 0].max().item()
                # new_proposals_idx_list.append(new_proposals_idx)
            cls_pred_list.append(cls_pred)
            score_pred_list.append(score_pred)
            mask_pred_list.extend([rle_encode(i) for i in mask_pred.cpu().numpy()])
        cls_pred = torch.cat(cls_pred_list).cpu().numpy()
        score_pred = torch.cat(score_pred_list).cpu().numpy()
        # mask_pred = torch.cat(mask_pred_list).cpu().numpy()
        # new_proposals_idx = torch.cat(new_proposals_idx_list, 0).cpu().numpy()
        # remapper = np.zeros(num_instances * self.instance_classes, dtype=np.int64)
        # remapper[np.unique(new_proposals_idx[..., 0])] = np.arange(mask_pred.shape[0])
        # new_proposals_idx[..., 0] = remapper[new_proposals_idx[..., 0]]

        instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_id
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = mask_pred_list[i]  # rle_encode(mask_pred[i])
            instances.append(pred)
        return instances, score_pred  #, new_proposals_idx

    def get_gt_instances(self, semantic_labels, instance_labels):
        """Get gt instances for evaluation."""
        # convert to evaluation format 0: ignore, 1->N: valid
        # import ipdb; ipdb.set_trace(context=10)
        # label_shift = self.semantic_classes - self.instance_classes
        semantic_labels = semantic_labels - self.label_shift + 1
        semantic_labels[semantic_labels < 0] = 0
        instance_labels += 1
        ignore_inds = instance_labels < 0
        # scannet encoding rule
        gt_ins = semantic_labels * 1000 + instance_labels
        gt_ins[ignore_inds] = 0
        gt_ins = gt_ins.cpu().numpy()
        return gt_ins

    def panoptic_fusion(self, semantic_preds, instance_preds):
        cls_offset = self.label_shift - 1
        panoptic_cls = semantic_preds.copy().astype(np.uint32)
        panoptic_ids = np.zeros_like(semantic_preds).astype(np.uint32)

        # higher score has higher fusion priority
        scores = [x['conf'] for x in instance_preds]
        score_inds = np.argsort(scores)[::-1]
        prev_paste = np.zeros_like(semantic_preds, dtype=bool)
        panoptic_id = 1
        for i in score_inds:
            instance = instance_preds[i]
            cls = instance['label_id']  # [1 ~ num_inst_classes]
            mask = rle_decode(instance['pred_mask']).astype(bool)

            # check overlap with pasted instances
            intersect = (mask * prev_paste).sum()
            if intersect / (mask.sum() + 1e-5) > self.test_cfg.PANOPTIC_EVAL.SKIP_IOU:
                continue

            paste = mask * (~prev_paste)
            panoptic_cls[paste] = cls + cls_offset  # real sem id
            panoptic_ids[paste] = panoptic_id
            prev_paste[paste] = 1
            panoptic_id += 1

        # if thing classes have panoptic id == 0, ignore it
        ignore_inds = (panoptic_cls >= self.inst_class_idx.min()) & \
            (panoptic_cls <= self.inst_class_idx.max()) & (panoptic_ids == 0)

        # encode panoptic results
        panoptic_preds = (panoptic_cls & 0xFFFF) | (panoptic_ids << 16)
        panoptic_preds[ignore_inds] = len(self.valid_class_idx)
        panoptic_preds = panoptic_preds.astype(np.uint32)
        return panoptic_preds

    def forward(self, batch_dict):
        self.forward_ret_dict = {}
        self.semantic_only = batch_dict['epoch'] < self.prepare_epoch

        if not self.training and batch_dict['test_x4_split']:
            batch_dict['points_xyz'] = common_utils.merge_4_parts(batch_dict['points_xyz'])
            batch_dict['labels'] = common_utils.merge_4_parts(batch_dict['labels'])
            batch_dict['inst_label'] = common_utils.merge_4_parts(batch_dict['inst_label'])
            batch_dict['pt_offset_label'] = common_utils.merge_4_parts(batch_dict['pt_offset_label'])
            batch_dict['binary_labels'] = common_utils.merge_4_parts(batch_dict['binary_labels'])
            self.forward_ret_dict['seg_labels'] = batch_dict['labels']
            self.forward_ret_dict['binary_labels'] = batch_dict['binary_labels']

        # forward offset_branch
        backbone3d_feats = batch_dict[self.in_feature_name]
        if not self.no_v2p_map:
            backbone3d_feats = backbone3d_feats[batch_dict['v2p_map']]
        if not self.training and batch_dict['test_x4_split']:
            backbone3d_feats = common_utils.merge_4_parts(backbone3d_feats)
        pt_offsets = self.offset_linear(backbone3d_feats)
        # pt_offsets[batch_dict['inst_label'] == self.ignore_label] = -100.
        # pt_offsets = batch_dict['pt_offset_label'] * 0.2
        # pt_offsets[batch_dict['inst_label'] == self.ignore_label] = 0.

        # === load offset from file ===
        # offset_list = []
        # for s in batch_dict['scene_name']:
        #     offset = np.load(f'../output/scannet_models/inst/softgroup_clip_base13_caption_adamw/default/eval/epoch_150/val/proposal/offset_pred/{s.item()}.npy')
        #     offset_list.append(offset)
        # pt_offsets_offline = torch.from_numpy(np.concatenate(offset_list, 0)).cuda() * 0.5
        # pt_offsets[(pt_offsets_offline == 0).sum(1) < 3] = pt_offsets_offline[(pt_offsets_offline == 0).sum(1) < 3]
        # ==============================
        self.forward_ret_dict['pt_offsets'] = pt_offsets
        self.forward_ret_dict['pt_offset_label'] = batch_dict['pt_offset_label']
        self.forward_ret_dict['inst_label'] = batch_dict['inst_label']
        self.forward_ret_dict['pt_offset_mask'] = batch_dict.get('pt_offset_mask', None)

        # forward top-down module
        semantic_scores, batch_idxs, points, batch_size = \
            batch_dict['seg_scores'], batch_dict['batch_idxs'], batch_dict['points_xyz'], \
            batch_dict['batch_size']
        binary_scores = batch_dict['binary_ret_dict']['binary_scores'] if 'binary_ret_dict' in batch_dict else None

        if not self.semantic_only:
            if batch_dict.get('gt_label_generation', False):
                valid_idx = (torch.abs(batch_dict['pt_offset_label']) < 60).sum(1) == 3
                self.forward_ret_dict['pt_offsets'][valid_idx] = batch_dict['pt_offset_label'][valid_idx] * 0.5
                return batch_dict  # early return
            elif batch_dict.get('pseudo_label_generation', False):
                proposals_idx, proposals_offset, cls_scores, proposal_binary_scores = self.forward_grouping(
                    batch_size, semantic_scores, pt_offsets, batch_idxs, points,
                    super_voxel=batch_dict.get('super_voxel', None), output_feats=backbone3d_feats,
                    binary_scores=binary_scores, binary_labels=batch_dict['binary_labels'],
                    pseudo_label_generation=True)
                # === obtain proposal center =============
                if len(proposals_idx) > 0:
                    proposals_center = scatter_mean(points.cpu()[proposals_idx[..., 1].long()], index=proposals_idx[..., 0].long(), dim=0)
                    points_center = proposals_center[proposals_idx[..., 0].long()]
                    points_center = scatter_mean(points_center, index=proposals_idx[..., 1].long(), dim=0)  # (N, C)
                    if points_center.shape[0] < points.shape[0]:
                        remain_points = points.shape[0] - points_center.shape[0]
                        points_center = torch.cat((points_center, torch.zeros((remain_points, 3))), 0)
                    point_offset = points_center - points.cpu()
                    point_offset[(points_center == 0).sum(1) == 3] = 0.
                else:
                    point_offset = points.cpu().clone()
                    point_offset[:] = 0.
                # pt_offsets = point_offset
                self.forward_ret_dict['pt_offsets'] = point_offset
                # self.forward_ret_dict['pt_offsets'] = self.forward_ret_dict['pt_offset_label']
                # ========================================
                return batch_dict  # early return
            else:
                proposals_idx, proposals_offset, cls_scores, proposal_binary_scores = self.forward_grouping(
                    batch_size, semantic_scores, pt_offsets, batch_idxs, points,
                    binary_scores=binary_scores)
            if not self.training and len(proposals_idx) == 0:  # no proposal during validation
                self.forward_ret_dict['no_proposal'] = True
                return batch_dict
            if self.training and len(proposals_idx) == 0:
                point_nth_multiplier = 1.0
                while len(proposals_idx) == 0 and point_nth_multiplier > 0:
                    print(batch_dict['ids'], 'no proposal!')
                    point_nth_multiplier /= 2.
                    proposals_idx, proposals_offset, cls_scores, proposal_binary_scores = self.forward_grouping(
                        batch_size, semantic_scores, pt_offsets, batch_idxs, points,
                        binary_scores=binary_scores, point_nth_multiplier=point_nth_multiplier)
                if len(proposals_idx) == 0:  # still no proposals
                    return batch_dict
                # print(proposals_offset.shape[0])
                if self.training and proposals_offset.shape[0] > self.loss_cfg.MAX_PROPOSAL_NUM:
                    proposals_offset = proposals_offset[:self.loss_cfg.MAX_PROPOSAL_NUM + 1]
                    proposals_idx = proposals_idx[:proposals_offset[-1]]
                    cls_scores = cls_scores[:self.loss_cfg.MAX_PROPOSAL_NUM]
                    assert proposals_idx.shape[0] == proposals_offset[-1]
            inst_feats, inst_map = self.clusters_voxelization(
                proposals_idx, proposals_offset, backbone3d_feats,
                points, rand_quantize=self.training, **self.inst_voxel_cfg)

            inst_batch_idxs, iou_scores, mask_scores = self.forward_instance(
                inst_feats, inst_map)
            if self.training:
                self.forward_ret_dict['mask_scores'] = mask_scores
                self.forward_ret_dict['iou_scores'] = iou_scores
                self.forward_ret_dict['proposals_idx'] = proposals_idx
                self.forward_ret_dict['proposals_offset'] = proposals_offset
                self.forward_ret_dict['inst_pointnum'] = batch_dict['inst_pointnum']
                self.forward_ret_dict['inst_cls'] = batch_dict['inst_cls']
                self.forward_ret_dict['inst_batch_idxs'] = inst_batch_idxs

            if not self.training:
                pred_instances, _ = self.get_instances(
                    batch_dict['ids'][0], proposals_idx, semantic_scores, cls_scores, iou_scores,
                    mask_scores, proposal_binary_scores=proposal_binary_scores, binary_scores=binary_scores)
                if 'instance' in self.test_cfg.EVAL_TASKS:
                    gt_instances = self.get_gt_instances(batch_dict['labels'].clone(), batch_dict['inst_label'].clone())
                    self.forward_ret_dict['pred_instances'] = pred_instances
                    self.forward_ret_dict['gt_instances'] = gt_instances
                if 'panoptic' in self.test_cfg.EVAL_TASKS:
                    panoptic_preds = self.panoptic_fusion(batch_dict['seg_preds'].cpu().numpy(), pred_instances)
                    self.forward_ret_dict['panoptic_preds'] = panoptic_preds
        #     ret_dict.update({'mask_scores': mask_scores, 'iou_scores': iou_scores,
        #         'proposals_idx': proposals_idx, 'proposals_offset': proposals_offset,
        #         'instance_batch_idxs': instance_batch_idxs, 'cls_scores': proposals_sem_score})

        return batch_dict

    def get_offset_loss(self, pt_offsets, pt_offset_labels, inst_labels, pos_inds=None):
        # assert (pos_inds & (inst_labels != self.ignore_label)).sum().item() == (inst_labels != self.ignore_label).sum().item()
        # assert (pos_inds | (inst_labels != self.ignore_label)).sum().item() == pos_inds.sum().item()
        if pos_inds is None:
            pos_inds = (inst_labels != self.ignore_label)
        if pos_inds.sum() == 0:
            offset_loss = 0 * pt_offsets.sum()
        else:
            if self.offset_loss_type == 'l1':
                offset_loss = torch.nn.functional.l1_loss(
                    pt_offsets[pos_inds], pt_offset_labels[pos_inds], reduction='sum') / pos_inds.sum()
            elif self.offset_loss_type == 'cos':
                offset_loss = loss_utils.CosineSimilarityLoss()(pt_offsets, pt_offset_labels[pos_inds], pos_inds)
        return offset_loss

    @force_fp32(apply_to=('cls_scores', 'mask_scores', 'iou_scores'))
    def get_inst_loss(self, mask_scores, iou_scores, proposals_idx, proposals_offset, inst_labels,
                      inst_pointnum, inst_cls):
        # losses = {}
        _inst_labels = inst_labels.clone()
        _inst_labels[inst_labels == self.ignore_label] = -100
        proposals_idx = proposals_idx.cuda()
        proposals_offset = proposals_offset.cuda()

        # cal iou of clustered instance
        # import ipdb; ipdb.set_trace(context=10)
        ious_on_cluster = ops.get_mask_iou_on_cluster(
            proposals_idx[:, 1].contiguous(), proposals_offset, _inst_labels, inst_pointnum)

        # filter out background instances
        fg_inds = (inst_cls != self.ignore_label)
        fg_instance_cls = inst_cls[fg_inds]
        fg_ious_on_cluster = ious_on_cluster[:, fg_inds]

        # assign proposal to gt idx. -1: negative, 0 -> num_gts - 1: positive
        num_proposals = fg_ious_on_cluster.size(0)
        num_gts = fg_ious_on_cluster.size(1)
        if num_proposals == 0 or num_gts == 0:
            # losses.update({'mask_loss': mask_scores.sum() * 0.0, 'iou_score_loss': iou_scores.sum() * 0.0})
            return mask_scores.sum() * 0.0, iou_scores.sum() * 0.0
        assigned_gt_inds = fg_ious_on_cluster.new_full((num_proposals, ), -1, dtype=torch.long)

        # overlap > thr on fg instances are positive samples
        max_iou, argmax_iou = fg_ious_on_cluster.max(1)
        pos_inds = max_iou >= self.loss_cfg.POS_IOU_THR
        assigned_gt_inds[pos_inds] = argmax_iou[pos_inds]

        # allow low-quality proposals with best iou to be as positive sample
        # in case pos_iou_thr is too high to achieve
        match_low_quality = getattr(self.loss_cfg, 'MATCH_LOW_QUALITY', False)
        min_pos_thr = getattr(self.loss_cfg, 'MIN_POS_THR', 0)
        if match_low_quality:
            gt_max_iou, gt_argmax_iou = fg_ious_on_cluster.max(0)
            for i in range(num_gts):
                if gt_max_iou[i] >= min_pos_thr:
                    assigned_gt_inds[gt_argmax_iou[i]] = i

        # compute cls loss. follow detection convention: 0 -> K - 1 are fg, K is bg
        labels = fg_instance_cls.new_full((num_proposals, ), len(self.inst_class_idx))
        pos_inds = assigned_gt_inds >= 0
        labels[pos_inds] = fg_instance_cls[assigned_gt_inds[pos_inds]]
        # cls_loss = F.cross_entropy(cls_scores[pos_inds], labels[pos_inds])
        # losses['cls_loss'] = cls_loss

        # compute mask loss
        # mask_cls_label = labels[instance_batch_idxs.long()]
        # slice_inds = torch.arange(
        #     0, mask_cls_label.size(0), dtype=torch.long, device=mask_cls_label.device)
        # mask_scores_sigmoid_slice = mask_scores.sigmoid()[slice_inds, mask_cls_label]
        mask_scores_sigmoid_slice = mask_scores.sigmoid().reshape(-1)
        mask_label = ops.get_mask_label(proposals_idx[:, 1].contiguous(), proposals_offset, _inst_labels,
            inst_cls, inst_pointnum, ious_on_cluster, self.loss_cfg.POS_IOU_THR)
        mask_label_weight = (mask_label != -1).float()
        mask_label[mask_label == -1.] = 0.5  # any value is ok
        mask_loss = torch.nn.functional.binary_cross_entropy(
            mask_scores_sigmoid_slice, mask_label, weight=mask_label_weight, reduction='sum')
        mask_loss /= (mask_label_weight.sum() + 1)
        # losses['mask_loss'] = mask_loss

        # compute iou score loss
        ious = ops.get_mask_iou_on_pred(proposals_idx[:, 1].contiguous(), proposals_offset, _inst_labels,
            inst_pointnum, mask_scores_sigmoid_slice.detach())
        fg_ious = ious[:, fg_inds]
        gt_ious, _ = fg_ious.max(1)
        # slice_inds = torch.arange(0, labels.size(0), dtype=torch.long, device=labels.device)
        iou_score_weight = (labels < len(self.inst_class_idx)).float()
        # iou_score_slice = iou_scores[slice_inds, labels]
        iou_score_slice = iou_scores.reshape(-1)
        iou_score_loss = torch.nn.functional.mse_loss(iou_score_slice, gt_ious, reduction='none')
        iou_score_loss = (iou_score_loss * iou_score_weight).sum() / (iou_score_weight.sum() + 1)
        # losses['iou_score_loss'] = iou_score_loss
        return mask_loss, iou_score_loss

    def get_loss(self):
        pt_offsets = self.forward_ret_dict['pt_offsets']
        pt_offset_labels = self.forward_ret_dict['pt_offset_label']
        inst_labels = self.forward_ret_dict['inst_label']
        pt_offset_mask = self.forward_ret_dict['pt_offset_mask']

        loss = self.get_offset_loss(pt_offsets, pt_offset_labels, inst_labels, pt_offset_mask)
        tb_dict = {'offset_loss': loss.item()}

        if not self.semantic_only and 'proposals_offset' in self.forward_ret_dict:
            mask_scores = self.forward_ret_dict['mask_scores']
            iou_scores = self.forward_ret_dict['iou_scores']
            proposals_idx = self.forward_ret_dict['proposals_idx']
            proposals_offset = self.forward_ret_dict['proposals_offset']
            inst_cls = self.forward_ret_dict['inst_cls']
            inst_pointnum = self.forward_ret_dict['inst_pointnum']
            mask_loss, iou_score_loss = self.get_inst_loss(
                mask_scores, iou_scores, proposals_idx, proposals_offset, inst_labels,
                inst_pointnum, inst_cls)
            loss += (mask_loss + iou_score_loss)
            tb_dict.update({'mask_loss': mask_loss.item(), 'iou_score_loss': iou_score_loss.item()})
        return loss, tb_dict

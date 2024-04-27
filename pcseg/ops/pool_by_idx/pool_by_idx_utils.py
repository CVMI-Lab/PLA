import torch
from torch.autograd import Function

from . import pool_by_idx


class AvgPoolByIdx(Function):

    @staticmethod
    def forward(ctx, feats, point_idx, caption_idx, caption_idx_offset, n_cap_per_point, base_mask):
        """
        Args:
            ctx:
            feats: (N, C) float
            point_idx: (N0) long, ranging from -1 to N
            caption_idx: [(M,)] [long], length=K, ranging from 0 to N0
            caption_idx_offset:
            n_cap_per_point:
            base_mask:

        Returns:
            pooled_feats: (K, C) float
            real_n_points: (K, ) bool

        """
        n_captions = len(caption_idx_offset) - 1
        n_pts, n_channel = feats.size()
        assert feats.is_contiguous()
        assert point_idx.is_contiguous()

        pooled_feats = torch.cuda.FloatTensor(n_captions, n_channel).zero_()
        real_n_points = torch.cuda.LongTensor(n_captions).zero_()
        pool_by_idx.avg_pool_by_idx_fp(
            feats, pooled_feats, point_idx, caption_idx,
            caption_idx_offset, real_n_points, n_captions, n_channel
        )

        # if_has_pts = real_n_points > 0
        ctx.for_backwards = (n_pts, real_n_points, point_idx, caption_idx, caption_idx_offset, n_cap_per_point, base_mask)

        return pooled_feats, real_n_points

    @staticmethod
    def backward(ctx, d_output_feats, d_if_has_pts):
        n_captions, n_channel = d_output_feats.size()

        n_pts, real_n_points, point_idx, caption_idx, caption_idx_offset, n_cap_per_point, base_mask = ctx.for_backwards
        d_feats = torch.cuda.FloatTensor(n_pts, n_channel).zero_()

        pool_by_idx.avg_pool_by_idx_bp(
            d_feats, d_output_feats.contiguous(), point_idx,
            caption_idx, caption_idx_offset, real_n_points, n_captions, n_channel)

        if n_cap_per_point is not None:
            # to make sure no divide 0 error
            n_cap_per_point = torch.clamp(n_cap_per_point, min=1.0).unsqueeze(1)
            # debug code
            # d_feats_norm = torch.norm(d_feats, dim=1)
            # print(d_feats_norm)
            # print(n_cap_per_point[:, 0])
            # print(n_cap_per_point[:, 0].max(), n_cap_per_point[:, 0].argmax())

            d_feats = d_feats / n_cap_per_point

        d_feats[base_mask] = 0.
        return d_feats, None, None, None, None, None


avg_pool_by_idx = AvgPoolByIdx.apply

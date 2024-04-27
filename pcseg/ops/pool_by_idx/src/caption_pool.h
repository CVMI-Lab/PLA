/*
Caption Avg Pool by Idx
Written by Runyu Ding
All Rights Reserved 2022.
*/

#ifndef CAPTION_POOL_H
#define CAPTION_POOL_H
#include <ATen/cuda/CUDAContext.h>
#include <torch/serialize/tensor.h>

#include "../datatype/datatype.h"

void avg_pool_by_idx_fp_cuda(float *feats, float *output_feats,
                             long *point_idx, long *caption_idx,
                             long *caption_idx_offset, long *real_n_points,
                             int n_captions, int C);

void avg_pool_by_idx_bp_cuda(float *d_feats, float *d_output_feats,
                             long *point_idx, long *caption_idx,
                             long *caption_idx_offset, long *real_n_points,
                             int n_captions, int C);

void avg_pool_by_idx_fp(at::Tensor feats_tensor,
                        at::Tensor output_feats_tensor,
                        at::Tensor point_idx_tensor,
                        at::Tensor caption_idx_tensor,
                        at::Tensor caption_idx_offset_tensor,
                        at::Tensor real_n_points_tensor,
                        int n_captions, int C);

void avg_pool_by_idx_bp(at::Tensor d_feats_tensor,
                        at::Tensor d_output_feats_tensor,
                        at::Tensor point_idx_tensor,
                        at::Tensor caption_idx_tensor,
                        at::Tensor caption_idx_offset_tensor,
                        at::Tensor real_n_points_tensor,
                        int n_captions, int C);

#endif // CAPTION_POOL_H

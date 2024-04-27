/*
Avg Pool by Idx
Written by Runyu Ding
All Rights Reserved 2022.
*/

#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <assert.h>

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
                        int n_captions, int C) {
  float *feats = feats_tensor.data_ptr<float>();
  float *output_feats = output_feats_tensor.data_ptr<float>();
  long *point_idx = point_idx_tensor.data_ptr<long>();
  long *caption_idx = caption_idx_tensor.data_ptr<long>();
  long *caption_idx_offset = caption_idx_offset_tensor.data_ptr<long>();
  long *real_n_points = real_n_points_tensor.data_ptr<long>();

  avg_pool_by_idx_fp_cuda(
    feats, output_feats, point_idx, caption_idx,
    caption_idx_offset, real_n_points, n_captions, C);
}

void avg_pool_by_idx_bp(at::Tensor d_feats_tensor,
                        at::Tensor d_output_feats_tensor,
                        at::Tensor point_idx_tensor,
                        at::Tensor caption_idx_tensor,
                        at::Tensor caption_idx_offset_tensor,
                        at::Tensor real_n_points_tensor,
                        int n_captions, int C) {
  float *d_feats = d_feats_tensor.data_ptr<float>();
  float *d_output_feats = d_output_feats_tensor.data_ptr<float>();
  long *point_idx = point_idx_tensor.data_ptr<long>();
  long *caption_idx = caption_idx_tensor.data_ptr<long>();
  long *caption_idx_offset = caption_idx_offset_tensor.data_ptr<long>();
  long *real_n_points = real_n_points_tensor.data_ptr<long>();

  avg_pool_by_idx_bp_cuda(
    d_feats, d_output_feats, point_idx, caption_idx, caption_idx_offset,
    real_n_points, n_captions, C);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("avg_pool_by_idx_fp", &avg_pool_by_idx_fp, "avg_pool_by_idx_fp");
    m.def("avg_pool_by_idx_bp", &avg_pool_by_idx_bp, "avg_pool_by_idx_bp");
}
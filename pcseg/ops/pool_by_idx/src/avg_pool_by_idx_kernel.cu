/*
Caption Avg Pool by Idx
Written by Runyu Ding
All Rights Reserved 2022.
*/

#include <math.h>
#include <stdio.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/serialize/tensor.h>

// fp
__global__ void avg_pool_by_idx_fp_cuda_(float *feats, float *output_feats,
                                         long *point_idx, long *caption_idx,
                                         long *caption_idx_offset, long *real_n_points,
                                         int n_captions, int C) {
  for (int pp_id = blockIdx.x; pp_id < n_captions; pp_id += gridDim.x) {
    int start = caption_idx_offset[pp_id];
    int end = caption_idx_offset[pp_id + 1];

    for (int plane = threadIdx.x; plane < C; plane += blockDim.x) {
      // int argmax_idx = -1;
      // float max_val = -1e50;
      float val = 0;
      int n_points = end - start;
      // int n_points2 = 0;

      for (int i = start; i < end; i++) {
        long idx = point_idx[caption_idx[i]];
        if (idx >= 0) {
          val += feats[idx * C + plane];
          // n_points2 += 1;
        } else {
          n_points -= 1;
        }
      }
      // output_maxidx[pp_id * C + plane] = argmax_idx;
      // printf("n_points: %d, %d", n_points, n_points2);
      if (plane == 0) {
        real_n_points[pp_id] = n_points;
      }
      if (n_points >= 0) {
        output_feats[pp_id * C + plane] = val / (float)n_points;
      }
    }
  }
}

// input: feats (sumNPoint, C) float
// input: proposals_offset (nProposal + 1) int
// output: output_feats (nProposal, C) float
// output: output_maxidx (nProposal, C) int
void avg_pool_by_idx_fp_cuda(float *feats, float *output_feats,
                             long *point_idx, long *caption_idx,
                             long *caption_idx_offset, long *real_n_points,
                             int n_captions, int C) {
  avg_pool_by_idx_fp_cuda_<<<std::min(n_captions, (int)1024),
                             std::min(C, (int)512)>>>(
      feats, output_feats, point_idx, caption_idx, caption_idx_offset,
      real_n_points, n_captions, C);
}

// bp
__global__ void avg_pool_by_idx_bp_cuda_(float *d_feats, float *d_output_feats,
                                         long *point_idx, long *caption_idx,
                                         long *caption_idx_offset, long *real_n_points,
                                         int n_captions, int C) {
  for (int pp_id = blockIdx.x; pp_id < n_captions; pp_id += gridDim.x) {
    int start = caption_idx_offset[pp_id];
    int end = caption_idx_offset[pp_id + 1];
    int n_points = real_n_points[pp_id];
    if (n_points > 0) {
      for (int plane = threadIdx.x; plane < C; plane += blockDim.x) {
        for (int i = start; i < end; i++) {
          long idx = point_idx[caption_idx[i]];
          if (idx >= 0) { 
            atomicAdd(&d_feats[idx * C + plane],
                      d_output_feats[pp_id * C + plane] / (float)n_points);
          }
        }
      }
    }
  }
}

// input: d_output_feats (nProposal, C) float
// input: output_maxidx (nProposal, C) int
// input: proposals_offset (nProposal + 1) int
// output: d_feats (sumNPoint, C) float
void avg_pool_by_idx_bp_cuda(float *d_feats, float *d_output_feats,
                             long *point_idx, long *caption_idx,
                             long *caption_idx_offset, long *real_n_points,
                             int n_captions, int C) {
  avg_pool_by_idx_bp_cuda_<<<std::min(n_captions, (int)1024),
                             std::min(C, (int)512)>>>(
      d_feats, d_output_feats, point_idx, caption_idx, caption_idx_offset,
      real_n_points, n_captions, C);
}

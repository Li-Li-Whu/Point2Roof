#ifndef _CLUSTER_GPU_H
#define _CLUSTER_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int dbscan_wrapper_fast(int b, int n, float eps, int min_pts, at::Tensor xyz_tensor, at::Tensor idx_tensor);

void dbscan_kernel_launcher_fast(int b, int n, float eps, int min_pts, const float *xyz, int *idx);

int cluster_pts_wrapper_fast(int b, int n, int m, at::Tensor xyz_tensor, at::Tensor idx_tensor,
    at::Tensor new_xyz_tensor, at::Tensor num_tensor);

void cluster_pts_kernel_launcher_fast(int b, int n, int m, const float *xyz, const int *idx, float *new_xyz, int *num);


int dbscan_wrapper_stack(int b, int n, float eps, int min_pts, at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor,
    at::Tensor idx_tensor);


void dbscan_kernel_launcher_stack(int b, int n, float eps, int min_pts,
    const float *xyz, const int *xyz_batch_cnt, int *idx);

int cluster_pts_wrapper_stack(int B, at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor, at::Tensor idx_tensor,
    at::Tensor new_xyz_tensor, at::Tensor cluster_cnt_tensor);


void cluster_pts_kernel_launcher_stack(int B, const float *xyz, const int *xyz_batch_cnt, int *idx,
    const float *new_xyz, const int *cluster_cnt);

#endif


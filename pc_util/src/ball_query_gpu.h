#ifndef _BALL_QUERY_GPU_H
#define _BALL_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int ball_query_wrapper_fast(int b, int n, int m, float radius, int nsample, 
	at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor);

void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, 
	const float *new_xyz, const float *xyz, int *idx);

int ball_center_query_wrapper_fast(int b, int n, int m, float radius,
    at::Tensor point_tensor, at::Tensor key_point_tensor, at::Tensor idx_tensor);

void ball_center_query_kernel_launcher_fast(int b, int n, int m, float radius,
    const float *point, const float *key_point, int *idx);

int knn_query_wrapper_fast(int b, int n, int m, int nsample,
	at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor dist2_tensor, at::Tensor idx_tensor);

void knn_query_kernel_launcher_fast(int b, int n, int m, int nsample,
	const float *new_xyz, const float *xyz, float *dist2, int *idx);


int ball_query_wrapper_stack(int B, int M, float radius, int nsample,
    at::Tensor new_xyz_tensor, at::Tensor new_xyz_batch_cnt_tensor,
    at::Tensor xyz_tensor, at::Tensor xyz_batch_cnt_tensor, at::Tensor idx_tensor);


void ball_query_kernel_launcher_stack(int B, int M, float radius, int nsample,
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx);



#endif

#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "group_points_gpu.h"


__global__ void group_points_grad_kernel_fast(int b, int c, int n, int npoints, int nsample, 
    const float *__restrict__ grad_out, const int *__restrict__ idx, float *__restrict__ grad_points) {
    // grad_out: (B, C, npoints, nsample)
    // idx: (B, npoints, nsample)
    // output:
    //      grad_points: (B, C, N)
    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int pt_idx = index / nsample;
    if (bs_idx >= b || c_idx >= c || pt_idx >= npoints) return;

    int sample_idx = index % nsample;
    grad_out += bs_idx * c * npoints * nsample + c_idx * npoints * nsample + pt_idx * nsample + sample_idx;
    idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx; 
    
    atomicAdd(grad_points + bs_idx * c * n + c_idx * n + idx[0] , grad_out[0]);
}

void group_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints, int nsample, 
    const float *grad_out, const int *idx, float *grad_points) {
    // grad_out: (B, C, npoints, nsample)
    // idx: (B, npoints, nsample)
    // output:
    //      grad_points: (B, C, N)
    cudaError_t err;
    dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    group_points_grad_kernel_fast<<<blocks, threads>>>(b, c, n, npoints, nsample, grad_out, idx, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void group_points_kernel_fast(int b, int c, int n, int npoints, int nsample, 
    const float *__restrict__ points, const int *__restrict__ idx, float *__restrict__ out) {
    // points: (B, C, N)
    // idx: (B, npoints, nsample)
    // output:
    //      out: (B, C, npoints, nsample)
    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int pt_idx = index / nsample;
    if (bs_idx >= b || c_idx >= c || pt_idx >= npoints) return;

    int sample_idx = index % nsample;

    idx += bs_idx * npoints * nsample + pt_idx * nsample + sample_idx; 
    int in_idx = bs_idx * c * n + c_idx * n + idx[0];
    int out_idx = bs_idx * c * npoints * nsample + c_idx * npoints * nsample + pt_idx * nsample + sample_idx;

    out[out_idx] = points[in_idx];
}


void group_points_kernel_launcher_fast(int b, int c, int n, int npoints, int nsample, 
    const float *points, const int *idx, float *out) {
    // points: (B, C, N)
    // idx: (B, npoints, nsample)
    // output:
    //      out: (B, C, npoints, nsample)
    cudaError_t err;
    dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    group_points_kernel_fast<<<blocks, threads>>>(b, c, n, npoints, nsample, points, idx, out);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void group_points_grad_kernel_stack(int B, int M, int C, int N, int nsample,
    const float *grad_out, const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt, float *grad_features) {
    // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    // :return:
    //     grad_features: (N1 + N2 ..., C) gradient of the features
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = index % nsample;
    int C_idx = (index / nsample) % C;
    int pt_idx = (index / nsample / C);

    if (pt_idx >= M || C_idx >= C || sample_idx >= nsample) return;

    int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += idx_batch_cnt[k];
        bs_idx = k;
    }

    int features_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) features_batch_start_idx += features_batch_cnt[k];

    grad_out += pt_idx * C * nsample + C_idx * nsample + sample_idx;
    idx += pt_idx * nsample + sample_idx;
    grad_features += (features_batch_start_idx + idx[0]) * C + C_idx;

    atomicAdd(grad_features, grad_out[0]);
}

void group_points_grad_kernel_launcher_stack(int B, int M, int C, int N, int nsample,
    const float *grad_out, const int *idx, const int *idx_batch_cnt, const int *features_batch_cnt, float *grad_features) {
    // :param grad_out: (M1 + M2 ..., C, nsample) tensor of the gradients of the output from forward
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    // :return:
    //     grad_features: (N1 + N2 ..., C) gradient of the features

    cudaError_t err;
    // dim3 blocks(DIVUP(npoints * nsample, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 blocks(DIVUP(M * C * nsample, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    group_points_grad_kernel_stack<<<blocks, threads>>>(B, M, C, N, nsample, grad_out, idx, idx_batch_cnt, features_batch_cnt, grad_features);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void group_points_kernel_stack(int B, int M, int C, int nsample,
    const float *features, const int *features_batch_cnt, const int *idx, const int *idx_batch_cnt, float *out) {
    // :param features: (N1 + N2 ..., C) tensor of features to group
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
    // :return:
    //     output: (M1 + M2, C, nsample) tensor
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int sample_idx = index % nsample;
    int C_idx = (index / nsample) % C;
    int pt_idx = (index / nsample / C);

    if (pt_idx >= M || C_idx >= C || sample_idx >= nsample) return;

    int bs_idx = 0, pt_cnt = idx_batch_cnt[0];
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += idx_batch_cnt[k];
        bs_idx = k;
    }

    int features_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) features_batch_start_idx += features_batch_cnt[k];
    features += features_batch_start_idx * C;

    idx += pt_idx * nsample + sample_idx;
    int in_idx = idx[0] * C + C_idx;
    int out_idx = pt_idx * C * nsample + C_idx * nsample + sample_idx;

    out[out_idx] = features[in_idx];
}


void group_points_kernel_launcher_stack(int B, int M, int C, int nsample,
    const float *features, const int *features_batch_cnt, const int *idx, const int *idx_batch_cnt, float *out) {
    // :param features: (N1 + N2 ..., C) tensor of features to group
    // :param features_batch_cnt: (batch_size) [N1 + N2 ...] tensor containing the indicies of features to group with
    // :param idx: (M1 + M2 ..., nsample) tensor containing the indicies of features to group with
    // :param idx_batch_cnt: (batch_size) [M1 + M2 ...] tensor containing the indicies of features to group with
    // :return:
    //     output: (M1 + M2, C, nsample) tensor

    cudaError_t err;
    dim3 blocks(DIVUP(M * C * nsample, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    group_points_kernel_stack<<<blocks, threads>>>(B, M, C, nsample, features, features_batch_cnt, idx, idx_batch_cnt, out);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

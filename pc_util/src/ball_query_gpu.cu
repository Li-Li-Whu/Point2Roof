#include <math.h>
#include <stdio.h>
#include <stdlib.h>


#include "ball_query_gpu.h"
#include "cuda_utils.h"


__global__ void ball_query_kernel_fast(int b, int n, int m, float radius, int nsample,
    const float *__restrict__ new_xyz, const float *__restrict__ xyz, int *__restrict__ idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    idx += bs_idx * m * nsample + pt_idx * nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2){
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
}


void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, \
    const float *new_xyz, const float *xyz, int *idx) {
    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, M, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_kernel_fast<<<blocks, threads>>>(b, n, m, radius, nsample, new_xyz, xyz, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__global__ void ball_center_query_kernel_fast(int b, int n, int m, float radius, \
    const float *__restrict__ point, const float *__restrict__ key_point, int *__restrict__ idx) {
    // key_point: (B, M, 3)
    // point: (B, N, 3)
    // output:
    //      idx: (B, N)
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= n) return;

    point += bs_idx * n * 3 + pt_idx * 3;
    key_point += bs_idx * m * 3;
    idx += bs_idx * n + pt_idx;

    float radius2 = radius * radius;
    float point_x = point[0];
    float point_y = point[1];
    float point_z = point[2];

    float bestd = 1e8;
    for (int k = 0; k < m; ++k) {
        float x = key_point[k * 3 + 0];
        float y = key_point[k * 3 + 1];
        float z = key_point[k * 3 + 2];
        if (((x + 1) * (x + 1) + (y + 1) * (y + 1) + (z + 1) * (z + 1)) < 1e-4) break;
        float d2 = (point_x - x) * (point_x - x) + (point_y - y) * (point_y - y) + (point_z - z) * (point_z - z);
        if (d2 < radius2 && d2 < bestd){
            idx[0] = k;
            bestd = d2;
        }
    }
}


void ball_center_query_kernel_launcher_fast(int b, int n, int m, float radius, \
    const float *point, const float *key_point, int *idx) {
    // point: (B, n, 3)
    // key_point: (B, m, 3)
    // output:
    //      idx: (B, n)

    cudaError_t err;

    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_center_query_kernel_fast<<<blocks, threads>>>(b, n, m, radius, point, key_point, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}





__global__ void knn_query_kernel_fast(int b, int n, int m, int nsample, const float *__restrict__ new_xyz,
    const float *__restrict__ xyz, float *__restrict__ dist2, int *__restrict__ idx) {

    // new_xyz: (B, M, 3)
    // xyz: (B, N, 3)
    // output:
    //      dist2: (B, M, nsample)
    //      idx: (B, M, nsample)

    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || pt_idx >= m) return;

    new_xyz += bs_idx * m * 3 + pt_idx * 3;
    xyz += bs_idx * n * 3;
    dist2 += bs_idx * m * nsample + pt_idx * nsample;
    idx += bs_idx * m * nsample + pt_idx * nsample;

    float nx = new_xyz[0];
    float ny = new_xyz[1];
    float nz = new_xyz[2];

    for (int i = 0; i < n; ++i) {
        float x = xyz[i * 3 + 0];
        float y = xyz[i * 3 + 1];
        float z = xyz[i * 3 + 2];
        float d2 = (nx - x) * (nx - x) + (ny - y) * (ny - y) + (nz - z) * (nz - z);
        if (d2 < dist2[nsample - 1]) {
            dist2[nsample - 1] = d2;
            idx[nsample - 1] = i;
            for (int j = nsample - 2; j >= 0; j--) {
                if (d2 < dist2[j]){
                    dist2[j + 1] = dist2[j];
                    dist2[j] = d2;
                    idx[j + 1] = idx[j];
                    idx[j] = i;
                }
            }
        }
    }
}


void knn_query_kernel_launcher_fast(int b, int n, int m, int nsample, \
    const float *new_xyz, const float *xyz, float *dist2, int *idx) {
    cudaError_t err;

    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    knn_query_kernel_fast<<<blocks, threads>>>(b, n, m, nsample, new_xyz, xyz, dist2, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}








__global__ void ball_query_kernel_stack(int B, int M, float radius, int nsample, \
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx) {
    // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // output:
    //      idx: (M, nsample)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= M) return;

    int bs_idx = 0, pt_cnt = new_xyz_batch_cnt[0];
    for (int k = 1; k < B; k++){
        if (pt_idx < pt_cnt) break;
        pt_cnt += new_xyz_batch_cnt[k];
        bs_idx = k;
    }

    int xyz_batch_start_idx = 0;
    for (int k = 0; k < bs_idx; k++) xyz_batch_start_idx += xyz_batch_cnt[k];
    // for (int k = 0; k < bs_idx; k++) new_xyz_batch_start_idx += new_xyz_batch_cnt[k];

    new_xyz += pt_idx * 3;
    xyz += xyz_batch_start_idx * 3;
    idx += pt_idx * nsample;

    float radius2 = radius * radius;
    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];
    int n = xyz_batch_cnt[bs_idx];

    int cnt = 0;
    for (int k = 0; k < n; ++k) {
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < radius2){
            if (cnt == 0){
                for (int l = 0; l < nsample; ++l) {
                    idx[l] = k;
                }
            }
            idx[cnt] = k;
            ++cnt;
            if (cnt >= nsample) break;
        }
    }
    if (cnt == 0) idx[0] = -1;
}


void ball_query_kernel_launcher_stack(int B, int M, float radius, int nsample,
    const float *new_xyz, const int *new_xyz_batch_cnt, const float *xyz, const int *xyz_batch_cnt, int *idx){
    // :param xyz: (N1 + N2 ..., 3) xyz coordinates of the features
    // :param xyz_batch_cnt: (batch_size), [N1, N2, ...]
    // :param new_xyz: (M1 + M2 ..., 3) centers of the ball query
    // :param new_xyz_batch_cnt: (batch_size), [M1, M2, ...]
    // output:
    //      idx: (M, nsample)

    cudaError_t err;

    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    ball_query_kernel_stack<<<blocks, threads>>>(B, M, radius, nsample, new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

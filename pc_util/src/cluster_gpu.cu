#include <math.h>
#include <stdio.h>
#include <stdlib.h>


#include "cluster_gpu.h"
#include "cuda_utils.h"


__device__ float get_dis(float x1, float y1, float z1, float x2, float y2, float z2) {
	float dis = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
	return sqrt(dis);
}
/*
__device__ void dfs (int i, int c, int n, int min_pts, const int* pts_cnt, const int* pts_adj, int* idx, int label) {
    idx[i] = c;
    if(pts_cnt[i] < min_pts) return;

    for(int j=0;j<n;j++) {

        int adj = pts_adj[i * n + j];
        printf("%d   %d     %d\n", i * n, i * n + j, adj);
        if (adj == -1) break;
        if (idx[adj] == -1)
            dfs(adj, c, n, min_pts, pts_cnt, pts_adj, idx, label);
    }
}
*/

__global__ void dbscan_kernel_fast(int b, int n, float eps, int min_pts, const float *__restrict__ xyz, int *__restrict__ idx,
    int *__restrict__ pts_cnt, int *__restrict__ pts_adj, int *__restrict__ pts_stack) {
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, N)
    int bs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b) return;

    xyz += bs_idx * n * 3;
    idx += bs_idx * n;
    pts_cnt += bs_idx * n;
    pts_stack += bs_idx * n;
    pts_adj += bs_idx * n * n;

    for(int i=0;i<n;i++) {
        pts_cnt[i] = 0;
        for(int j=0;j<n;j++) {
            pts_adj[i * n + j] = -1;
            if(i==j) continue;
            float x1 = xyz[i * 3 + 0];
            float y1 = xyz[i * 3 + 1];
            float z1 = xyz[i * 3 + 2];
            float x2 = xyz[j * 3 + 0];
            float y2 = xyz[j * 3 + 1];
            float z2 = xyz[j * 3 + 2];

            if(get_dis(x2, y2, z2, -10.0, -10.0, -10.0) < 1e-3) continue;
            if(get_dis(x1, y1, z1, x2, y2, z2) <= eps) {
            pts_adj[i * n + pts_cnt[i]] = j;
                pts_cnt[i] += 1;
            }

        }
    }

    int cluster_idx = 0;

    for(int i=0;i<n;i++) {
        if(idx[i] != -1) continue;

        if(pts_cnt[i] >= min_pts) {
            for(int j=0;j<n;j++)
                pts_stack[j] = -1;
            pts_stack[0] = i;
            int stack_idx = 0;
            int stack_len = 1;
            while (stack_idx < n && pts_stack[stack_idx] != -1)
            {
                int pts_idx = pts_stack[stack_idx];
                idx[pts_idx] = cluster_idx;
                if(pts_cnt[pts_idx] < min_pts){
                    stack_idx += 1;
                    continue;
                }
                for(int j=0;j<n;j++) {
                    int adj = pts_adj[pts_idx * n + j];
                    if (adj == -1) break;
                    if (idx[adj] == -1)
                    {
                        idx[adj] = -2;
                        pts_stack[stack_len++] = adj;
                    }
                }
                stack_idx += 1;
            }
            cluster_idx += 1;
        }
    }
}


void dbscan_kernel_launcher_fast(int b, int n, float eps, int min_pts, const float *xyz, int *idx) {
    // xyz: (B, N, 3)
    // output:
    //      idx: (B, N)

    cudaError_t err;

    dim3 blocks(DIVUP(b, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    int* pts_cnt;
    int* pts_stack;
	int* pts_adj;

	err = cudaMalloc((void**)&pts_cnt, b * n * sizeof(int));
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    err = cudaMalloc((void**)&pts_stack, b * n * sizeof(int));
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    err = cudaMalloc((void**)&pts_adj, b * n * n * sizeof(int));
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    dbscan_kernel_fast<<<blocks, threads>>>(b, n, eps, min_pts, xyz, idx, pts_cnt, pts_adj, pts_stack);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    cudaFree(pts_cnt);
    cudaFree(pts_stack);
    cudaFree(pts_adj);
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}



__global__ void cluster_pts_kernel_fast(int b, int n, int m, const float *__restrict__ xyz, const int *__restrict__ idx,
    float *__restrict__ new_xyz, int *__restrict__ num) {
    int bs_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b ) return;

    xyz += bs_idx * n * 3;
    idx += bs_idx * n;
    new_xyz += bs_idx * m * 3;
    num += bs_idx * m;

    for(int i=0;i<n;i++) {
        if (idx[i] == -1) continue;
        int c_idx = idx[i];
        new_xyz[c_idx * 3 + 0] += xyz[i * 3 + 0];
        new_xyz[c_idx * 3 + 1] += xyz[i * 3 + 1];
        new_xyz[c_idx * 3 + 2] += xyz[i * 3 + 2];
        num[c_idx] += 1;
    }
    for(int i=0;i<m;i++) {
        if (num[i] == 0) break;
        new_xyz[i * 3 + 0] /= num[i];
        new_xyz[i * 3 + 1] /= num[i];
        new_xyz[i * 3 + 2] /= num[i];
    }

}




void cluster_pts_kernel_launcher_fast(int b, int n, int m, const float *xyz, const int *idx, float *new_xyz, int *num) {
    cudaError_t err;

    dim3 blocks(DIVUP(b, THREADS_PER_BLOCK));  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    cluster_pts_kernel_fast<<<blocks, threads>>>(b, n, m, xyz, idx, new_xyz, num);
    // cudaDeviceSynchronize();  // for using printf in kernel function
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}



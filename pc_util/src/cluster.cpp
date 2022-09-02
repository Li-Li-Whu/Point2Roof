#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "cluster_gpu.h"

extern THCState *state;

#define CHECK_CUDA(x) do { \
	  if (!x.type().is_cuda()) { \
		      fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
		      exit(-1); \
		    } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
	  if (!x.is_contiguous()) { \
		      fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
		      exit(-1); \
		    } \
} while (0)
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int dbscan_wrapper_fast(int b, int n, float eps, int min_pts, at::Tensor xyz_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(xyz_tensor);
    const float *xyz = xyz_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    dbscan_kernel_launcher_fast(b, n, eps, min_pts, xyz, idx);
    return 1;
}


int cluster_pts_wrapper_fast(int b, int n, int m, at::Tensor xyz_tensor, at::Tensor idx_tensor,
    at::Tensor new_xyz_tensor, at::Tensor num_tensor) {
    CHECK_INPUT(xyz_tensor);
    CHECK_INPUT(idx_tensor);
    const float *xyz = xyz_tensor.data<float>();
    const int *idx = idx_tensor.data<int>();
    float *new_xyz = new_xyz_tensor.data<float>();
    int *num = num_tensor.data<int>();

    cluster_pts_kernel_launcher_fast(b, n, m, xyz, idx, new_xyz, num);
    return 1;
}


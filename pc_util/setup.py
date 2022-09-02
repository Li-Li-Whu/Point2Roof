from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pc_util',
    version='1.0',
    ext_modules=[
        CUDAExtension('pc_util', [
            'src/pointnet2_api.cpp',
            'src/ball_query.cpp',
            'src/ball_query_gpu.cu',
            'src/group_points.cpp',
            'src/group_points_gpu.cu',
            'src/interpolate.cpp',
            'src/interpolate_gpu.cu',
            'src/sampling.cpp',
            'src/sampling_gpu.cu',
            'src/cluster.cpp',
            'src/cluster_gpu.cu',
        ], extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)

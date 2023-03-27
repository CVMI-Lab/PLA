from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    setup(
        name='softgroup',
        version='1.0',
        description='SoftGroup: SoftGroup for 3D Instance Segmentation [CVPR 2022]',
        author='Thang Vu',
        author_email='thangvubk@kaist.ac.kr',
        # packages=['softgroup'],
        package_data={'ops': ['*/*.so']},
        ext_modules=[
            CUDAExtension(
                name='softgroup_ops',
                sources=[
                    'ops/src/softgroup_api.cpp', 'ops/src/softgroup_ops.cpp',
                    'ops/src/cuda.cu'
                ],
                extra_compile_args={
                    'cxx': ['-g'],
                    'nvcc': ['-O2']
                },
                include_dirs=['/data/anaconda3/envs/pt18/include/'])
        ],
        cmdclass={'build_ext': BuildExtension})

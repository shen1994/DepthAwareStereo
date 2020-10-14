import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='depthconv',
    ext_modules=[CUDAExtension(name='depthconv_cuda',
                      sources=['src/depthconv.c',
                               'src/depthconv_cuda.c',
                               'src/depthconv_cuda_kernel.cu'],
                      extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']}),
    ],
    cmdclass={'build_ext': BuildExtension})

if __name__ == '__main__':
    assert torch.cuda.is_available(), 'Please install CUDA for GPU support.'

import os
import torch
from setuptools import setup
from setuptools import find_packages
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension
from torch.utils.cpp_extension import load

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

this_file = os.path.dirname(os.path.realpath(__file__))

sources = [os.path.join(this_file, 'src/depthconv.cpp')]
headers = [get_pybind_include(), get_pybind_include(user=True)]
defines = []
with_cuda = False
build_extension = CppExtension
extra_compile_args = {"cxx": []}

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += [os.path.join(this_file, 'src/depthconv_cuda.cpp')]
    headers += [os.path.join(this_file, 'src/')]
    defines += [('WITH_CUDA', None)]
    with_cuda = True
    build_extension =  CUDAExtension
    extra_compile_args["cxx"] = ['-g']
    extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
    ]

extra_objects = ['src/depthconv_cuda_kernel.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ext_modules = [
    build_extension(
        '_depthconv_ext',
        include_dirs=headers,
        sources=sources,
        extra_objects=extra_objects,
        extra_compile_args=extra_compile_args
    )
]

setup(
    name='depthconv_ext',
    version='0.1.0',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)


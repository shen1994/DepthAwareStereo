cd src
nvcc -c -o depthconv_cuda_kernel.o depthconv_cuda_kernel.cu -x cu -Xcompiler -fPIC -std c++11 --gpu-architecture=compute_61 --gpu-code=sm_61 -DGPU -I/usr/local/cuda/include/ -DCUDNN --compiler-options "-DGPU -DCUDNN -fPIC"
cd ..
sudo python3 build.py install

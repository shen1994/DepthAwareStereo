cd src
nvcc -c -o depthavgpooling_cuda_kernel.o depthavgpooling_cuda_kernel.cu -x cu -Xcompiler -fPIC -std c++11 --gpu-architecture=compute_75 --gpu-code=sm_75 -DGPU -I/usr/local/cuda/include/ -DCUDNN --compiler-options "-DGPU -DCUDNN -fPIC"
cd ..
sudo python3 build.py install



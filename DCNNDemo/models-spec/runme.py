# -*- coding: utf-8 -*-
'''
Editor: westwell-gpu
Version: 1.0
Time: 2019/08/15
'''

import os

all_versions = [8.0, 9.0, 10.0, 10.1]
all_versions_cmd = ['61', # cuda8.0
                    '61', # cuda9.0
                    '75', # cuda10.0
                    '75'] # cuda10.1

if __name__ == "__main__":

    # get cuda version
    cuda_version = os.popen('nvcc -V')
    version_result = cuda_version.read()
    version_list = version_result.strip().split('\n')
    core_version = version_list[3]
    version_need = ''
    for i in range(len(core_version)):
        if core_version[i: i+7] == 'release':
            for j in range(i+8, len(core_version)):
                version_need += core_version[j]
                if core_version[j+1] == ',':
                    break
            break
    version_f = float(version_need)

    cuda_cmd = '61'
    for i in range(len(all_versions)):
        if version_f == all_versions[i]:
            cuda_cmd = all_versions_cmd[i]
            break

    current_path = os.path.abspath(os.path.dirname(__file__))

    # exec all cmds
    # os.system('rm -rf ' + current_path + '/ops/depthconv/_ext')
    # os.system('nvcc -c -o ' + current_path + '/ops/depthconv/src/depthconv_cuda_kernel.cu.o ' + \
    #     current_path + '/ops/depthconv/src/depthconv_cuda_kernel.cu ' + \
    #     '-x cu -Xcompiler -fPIC -std=c++11 -arch=sm_' + cuda_cmd)
    # os.system('CC=g++ python3 ' + current_path + '/ops/depthconv/build.py')

    # os.system('rm -rf ' + current_path + '/ops/depthavgpooling/_ext')
    # os.system('nvcc -c -o ' + current_path + '/ops/depthavgpooling/src/depthavgpooling_cuda_kernel.cu.o ' + \
    #     current_path + '/ops/depthavgpooling/src/depthavgpooling_cuda_kernel.cu ' + \
    #     '-x cu -Xcompiler -fPIC -std=c++11 -arch=sm_' + cuda_cmd)
    # os.system('CC=g++ python3 ' + current_path + '/ops/depthavgpooling/build.py')

    os.system('rm -rf ' + current_path + '/dcnnlib/pyboostcvconverter/build')
    os.system('mkdir ' + current_path + '/dcnnlib/pyboostcvconverter/build')
    os.system('cd ' + current_path + '/dcnnlib/pyboostcvconverter/build \ncmake .. \nmake -j8')

    os.system('rm -rf ' + current_path + '/dcnnlib/build')
    os.system('mkdir ' +  current_path + '/dcnnlib/build')
    os.system('cd ' + current_path + '/dcnnlib/build \ncmake .. \nmake -j8')


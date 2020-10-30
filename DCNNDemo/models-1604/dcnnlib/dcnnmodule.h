/*
 * Copyright 2019 <copyright holder> <email>
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

#ifndef DCNNMODULE_H
#define DCNNMODULE_H

#include <time.h>
#include <iostream>
#include <dirent.h>
#include <Python.h>
#include <pyboostcvconverter.hpp>
#include <opencv2/opencv.hpp>

class DCNNModule
{
public:
  DCNNModule(const char* _model_path = nullptr, const char* _config_path = nullptr, int _gpu_id = 0);
  ~DCNNModule();
  void DCNN_calc_addr();
  void DCNN_find_all_files(char*, int, std::vector<std::string>&, std::vector<std::string>&);
  void  DCNN_get_module(char*);
  char* DCNN_get_pythondir();
  char* DCNN_get_modeldir();
  char* DCNN_get_moduledir();
  char* DCNN_get_configdir();
  void DCNN_initialize();
  void DCNN_core_compute(const cv::Mat&, const cv::Mat&, cv::Mat&);
  void DCNN_release();
  
public:
  int gpu_id = 0;
  const int MAX_LENGTH = 512;
  char* python_dir = nullptr;
  char* model_dir = nullptr;
  char* config_dir = nullptr;
  char* module_dir = nullptr;
  PyObject* DCNNString;
  PyObject* DCNNImport;
  PyObject* DCNNDict;
  PyObject *DCNNObject;
  PyObject *model_params;
  PyObject *label_nc_i;
  PyObject *model_inputs;
  PyObject *image_i;
  PyObject *depth_i;
  npy_intp image_dims[3] = {500, 740, 3}; // image shape
  npy_intp depth_dims[2] = {500, 740}; // depth shape
};

#endif // DCNNMODULE_H

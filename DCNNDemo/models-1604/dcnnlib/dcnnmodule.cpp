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
#include "dcnnmodule.h"

DCNNModule::DCNNModule(const char* _model_path, const char* _config_path, int _gpu_id)
{
  python_dir = (char*)malloc(sizeof(char) * MAX_LENGTH);
  for(int i = 0; i < MAX_LENGTH; i++){
    python_dir[i] = 0;
  }
  model_dir = (char*)malloc(sizeof(char) * MAX_LENGTH);
  for(int i = 0; i < MAX_LENGTH; i++){
    model_dir[i] = 0;
  }
  config_dir = (char*)malloc(sizeof(char) * MAX_LENGTH);
  for(int i = 0; i < MAX_LENGTH; i++){
    config_dir[i] = 0;
  }
  module_dir = (char*)malloc(sizeof(char) * MAX_LENGTH);
  for(int i = 0; i < MAX_LENGTH; i++){
    module_dir[i] = 0;
  }
  
  // get model.pt path
  if(!(_model_path == nullptr)){
    for(int i = 0; i < strlen(_model_path); i++){
      model_dir[i] = _model_path[i];
    }
  }
  // get config.ini path
  if(!(_config_path == nullptr)){
    for(int i = 0; i < strlen(_config_path); i++){
      config_dir[i] = _config_path[i];
    }
  }
  
  if(_gpu_id != 0){
    gpu_id = _gpu_id;
  }
  
  DCNN_calc_addr();
}

DCNNModule::~DCNNModule()
{
  if(python_dir != nullptr){
    free(python_dir);
    python_dir = nullptr;
  }
  if(model_dir != nullptr){
    free(model_dir);
    model_dir = nullptr;
  }
  if(config_dir != nullptr){
    free(config_dir);
    config_dir = nullptr;
  }
  if(module_dir != nullptr){
    free(module_dir);
    module_dir = nullptr;
  }
}

void DCNNModule::DCNN_calc_addr()
{
  // get work root
  char *dir_buffer = (char*)malloc(sizeof(char) * MAX_LENGTH);
  for(int i = 0; i < MAX_LENGTH; i++){
  dir_buffer[i] = 0;
  }
  getcwd(dir_buffer, MAX_LENGTH);

  for(int i = strlen(dir_buffer)-1; i >= 0; i--){
    if(dir_buffer[i] == '/'){
      dir_buffer[i] = 0;
      break;
    }
    dir_buffer[i] = 0;
  }
  
  DCNNModule::DCNN_get_module(dir_buffer);
  
  // set python module dir
  strcpy(python_dir, "sys.path.append('");
  strcat(python_dir, module_dir);
  strcat(python_dir, "/");
  strcat(python_dir, "')");

  if(strlen(model_dir) == 0){
    // set model module dir
    strcpy(model_dir, dir_buffer);
    strcat(model_dir, "models/default.pt");
  }
  
  if(strlen(config_dir) == 0){
    // set model module dir
    strcpy(config_dir, dir_buffer);
    strcat(config_dir, "models/config.ini");
  }
  
  if(dir_buffer != nullptr){
    free(dir_buffer);
    dir_buffer = nullptr;
  }
}

void DCNNModule::DCNN_find_all_files(char *pPath, int nDeepth, std::vector<std::string>& files, std::vector<std::string>& files_name)
{
    DIR *pDir = NULL;
    struct dirent *pSTDirEntry;
    char *pChild = NULL;
    
    if ((pDir = opendir(pPath)) == NULL)
    {
      return ;
    }
    else
    {
      while ((pSTDirEntry = readdir(pDir)) != NULL)
      {
	if ((strcmp(pSTDirEntry->d_name, ".") == 0) || (strcmp(pSTDirEntry->d_name, "..") == 0))
        {
	  continue;
        }
        else
        {
	  if (pSTDirEntry->d_type & DT_DIR)
          {
	    pChild = (char*)malloc(sizeof(char) * (NAME_MAX + 1));
            if (pChild == NULL)
            {
	      perror("memory not enough.");
              return ;
            }
            memset(pChild, 0, NAME_MAX + 1);
            strcpy(pChild, pPath);
	    strcat(pChild, "/");
	    strcat(pChild, pSTDirEntry->d_name);
	    files.push_back(pChild);
	    files_name.push_back(pSTDirEntry->d_name);
            DCNNModule::DCNN_find_all_files(pChild, nDeepth + 1, files, files_name);
            free(pChild);
            pChild = NULL;
	  }
        }
      }
      closedir(pDir);
    }
}

void DCNNModule::DCNN_get_module(char* root)
{
  std::vector<std::string> files;
  std::vector<std::string> files_name;
  DCNNModule::DCNN_find_all_files(root, 0, files, files_name);
  
  for(int i=0; i < files_name.size(); i++){
    if(strcmp(files_name.at(i).c_str(), "models") == 0){
      memcpy(module_dir, files.at(i).c_str(), strlen(files.at(i).c_str())*sizeof(char));
    }
  }
}

char* DCNNModule::DCNN_get_pythondir()
{
  if(python_dir != nullptr)
    return python_dir;
}

char* DCNNModule::DCNN_get_modeldir()
{
  if(model_dir != nullptr)
    return model_dir;
}

char* DCNNModule::DCNN_get_configdir()
{
  if(config_dir != nullptr)
    return config_dir;
}

char* DCNNModule::DCNN_get_moduledir()
{
  if(module_dir != nullptr)
    return module_dir;
}

void DCNNModule::DCNN_initialize()
{
    Py_Initialize();
    
    _import_array();
    
    if (!Py_IsInitialized()) {  
      return; 
    }
  
    PyRun_SimpleString("import sys");  
    PyRun_SimpleString(python_dir); 
    // load pyscript
    DCNNString = PyUnicode_FromFormat("module");
    DCNNImport = PyImport_Import(DCNNString);
    if(!DCNNImport){
      std::cout<<"can't find module.py or import error"<<std::endl;
      return;
    }
    DCNNDict = PyModule_GetDict(DCNNImport);
    if(!DCNNDict){
      std::cout<<"can't load module dict"<<std::endl;
      return;
    }
    
    // load model
    DCNNObject = PyDict_GetItemString(DCNNDict, "load_model");
    
    model_params = PyTuple_New(3);
    PyTuple_SetItem(model_params, 0, Py_BuildValue("i", gpu_id));
    PyTuple_SetItem(model_params, 1, Py_BuildValue("s", model_dir));
    PyTuple_SetItem(model_params, 2, Py_BuildValue("s", config_dir));
    PyObject_CallObject(DCNNObject, model_params);
    
    // define variables which need to use
    DCNNObject = PyDict_GetItemString(DCNNDict, "forward");
    model_inputs = PyTuple_New(2);
    
    std::cout<<"westwell: DCNN Segmentation Model has loaded."<<std::endl;
}

void DCNNModule::DCNN_core_compute(const cv::Mat& image, const cv::Mat& depth, cv::Mat& segout)
{
    image_dims[0] = image.rows;
    image_dims[1] = image.cols;
    depth_dims[0] = depth.rows;
    depth_dims[1] = depth.cols;

    if(image_dims[0] != depth_dims[0] || image_dims[1] != depth_dims[1])
        return;
    
    image_i = PyArray_SimpleNewFromData(3, image_dims, NPY_UBYTE, image.data);
    depth_i = PyArray_SimpleNewFromData(2, depth_dims, NPY_UBYTE, depth.data);
    PyTuple_SetItem(model_inputs, 0, image_i);
    PyTuple_SetItem(model_inputs, 1, depth_i);

    PyObject *pyResult = nullptr;
    pyResult = PyObject_CallObject(DCNNObject, model_inputs);

    memcpy(segout.data,  (unsigned char*)PyArray_DATA((PyArrayObject*)pyResult), 
           image_dims[0] * image_dims[1] * sizeof(unsigned char));

    if(pyResult != nullptr){
        Py_DECREF(pyResult);
        pyResult = nullptr;       
    }
}

void DCNNModule::DCNN_release()
{   
    // release compute reources
    Py_DECREF(model_inputs);
    Py_DECREF(model_params);
      
    // release resources
    Py_DECREF(DCNNString);
    Py_DECREF(DCNNImport);
    Py_DECREF(DCNNDict);
    Py_DECREF(DCNNObject);
}

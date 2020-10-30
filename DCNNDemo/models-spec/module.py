import os
import torch
import copy
import configparser
import numpy as np
from TDeeplab import Deeplab_Solver

model = None

def load_model(_gpu_id, _model_path, _config_path):
  global model
  # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ['CUDA_VISIBLE_DEVICES'] = str(_gpu_id)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  config = configparser.ConfigParser()
  config.readfp(open(_config_path))
  _label_nc = int(config.get('model', 'label'))  
  model = copy.deepcopy(Deeplab_Solver(label_nc=_label_nc, model_path=_model_path))
  model.model.eval()

def forward(image, depth):
  global model
  segpred_numpy = model.forward(image, depth)
  
  return segpred_numpy.copy()

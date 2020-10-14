# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:46:21 2019

@author: shen1994
"""

import os
import argparse
import torch

from utils import util

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
        
    def initialize(self):
        # experiment specifics
        self.parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--depthconv', type=int, default=1, help='if specified, use depthconv')
        self.parser.add_argument('--depthglobalpool', type=int, default=1, help='if specified, use global pooling with depth')
        
        # input/output sizes
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--fineSize', type=str, default='400,700', help='then crop to this size')
        self.parser.add_argument('--label_nc', type=int, default=18, help='# of input image channels')
        
        # for setting inputs
        self.parser.add_argument('--list', type=str, default='dataset/train_list.txt', help='image and seg mask list file')
        self.parser.add_argument('--vallist', type=str, default='dataset/test_list.txt', help='image and seg mask list file')
        
        # for data argumentation:
        self.parser.add_argument('--flip', type=int, default=1, help='if specified, flip the images for data argumentation')
        self.parser.add_argument('--noise', type=int, default=0, help='if specified, flip the images for data argumentation')
        self.parser.add_argument('--scale', type=int, default=0, help='if specified, scale the images for data argumentation')
        self.parser.add_argument('--colorjitter', type=int, default=1, help='if specified, crop the images for data argumentation')
        self.parser.add_argument('--inputmode', default='bgr-mean', type=str, help='input image normalize option: bgr-mean, divstd-mean')
        self.parser.add_argument('--serial_batches',  action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        
        # for displays
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=100, help='frequency of saving the latest results')
        
        # train parameters
        self.parser.add_argument('--iterSize', type=int, default=10, help='# of iter at starting learning rate')
        self.parser.add_argument('--nepochs', type=int, default=5000, help='# of iter at starting learning rate')
        self.parser.add_argument('--lr', type=float, default=0.00025, help='initial learning rate for adam') # 0.00025
        self.parser.add_argument('--lr_power', type=float, default=0.9, help='power of learning rate policy')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum for sgd')
        self.parser.add_argument('--wd', type=float, default=0.0004, help='weight decay for sgd')
        
        # control
        self.parser.add_argument('--isTrain', type=int, default=1, help='control train and valid')
        self.parser.add_argument('--continue_train', type=int, default=1, help='continue training: load the latest model')

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            one_id = int(str_id)
            self.opt.gpu_ids.append(one_id)
                
        str_sizes = self.opt.fineSize.split(',')
        self.opt.fineSize = []
        for str_size in str_sizes:
            one_size = int(str_size)
            if one_size > 0:
                self.opt.fineSize.append(one_size)
                
        # set gpu ids
        torch.backends.cudnn.benchmark = False
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
            
        args = vars(self.opt)
        print('---------------- Options ----------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('---------------- End ----------------')

        if save:
            file_name = os.path.join(self.opt.checkpoints_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('---------------- Options ----------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('---------------- End ----------------\n')
                
        return self.opt
                
        
        

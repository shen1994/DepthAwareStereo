# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:59:36 2019

@author: shen1994
"""

import os
import cv2
import shutil
import numpy as np

if __name__ == "__main__":

    dataset_dir = 'dataset/'
    split_train = 8800

    # make data class
    if os.path.exists(dataset_dir + 'Edisparity'):
        shutil.rmtree(dataset_dir + 'Edisparity')
    os.mkdir(dataset_dir + 'Edisparity')

    # read color map
    c_counter = 0
    c_dict = dict()
    cverse_dict = dict()
    class_f = open(dataset_dir + 'class_colors.txt', 'r')
    lines = class_f.readlines()
    for i in range(0, len(lines), 2):
        c_key = lines[i].replace('\n', '')
        c_value = lines[i+1].replace('\n', '').split(' ')
        c_value = [int(e) for e in c_value]
        s_value, c_value_total = [], []
        for i in range(len(c_value)):
            if i % 3 == 0:
                s_value = []
            s_value.append(c_value[i])
            if i % 3 == 2:
                c_value_total.append(s_value)
        c_objects = [v for k, v in c_dict.items()]
        if c_key not in c_objects:
            c_counter += 1
        c_dict[c_counter - 1] = c_key
        for i in range(len(c_value) // 3):
            cverse_dict[tuple(c_value_total[i])] = c_counter - 1
    class_f.close()

    # write maps and mapsr for class
    map_f = open(dataset_dir + 'map.txt', 'w+')
    for key, value in c_dict.items():
        map_f.write(str(key)+'--->'+value+'\n')
    map_f.close()

    map_rf = open(dataset_dir + 'mapr.txt', 'w+')
    for key, value in cverse_dict.items():
        map_rf.write(str(value)+' '+str(key[0])+','+str(key[1])+','+str(key[2])+'\n')
    map_rf.close()

    # find depth max and make
    for onedir in os.listdir(dataset_dir + 'disparity'):
        depth_dir = dataset_dir + 'disparity/' + onedir
        depth = cv2.imread(depth_dir, 0)
        depth = np.uint8(depth / 127. * 255)
        cv2.imwrite(dataset_dir + 'Edisparity/' + onedir, depth)

    # split data for training
    counter = 0
    train_f = open(dataset_dir + 'train_list.txt', 'w+')
    test_f = open(dataset_dir + 'test_list.txt', 'w+')

    for onedir in os.listdir(dataset_dir + 'rgb'):
        rgb_dir = dataset_dir + 'rgb/' + onedir
        depth_dir = dataset_dir + 'Edisparity/' + onedir
        label_dir = dataset_dir + 'Elabel/' + onedir
        if counter < split_train:
            train_f.write(rgb_dir + ' ' + depth_dir + ' ' + label_dir + '\n')
        else:
            train_f.write(rgb_dir + ' ' + depth_dir + ' ' + label_dir + '\n')
            test_f.write(rgb_dir + ' ' + depth_dir + ' ' + label_dir + '\n')
        counter += 1

    train_f.close()
    test_f.close()

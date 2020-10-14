import os
import cv2
import time
import shutil
import torch
import numpy as np
from models.TDeeplab import Deeplab_Solver

if __name__ == "__main__":

    # select GPU for training
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    if os.path.exists('results'):
        shutil.rmtree('results')
    os.mkdir('results')
    os.mkdir('results/training')
    os.mkdir('results/testing')

    isTrans = True

    # define model
    model = Deeplab_Solver(label_nc=18,
                           model_path='./checkpoints/best_net_net.pt')
    model.model.eval()

    # define label to RGB
    mapr_dict = dict()
    mapr = open('dataset/mapr.txt', 'r')
    mapr_list = mapr.readlines()
    for i in range(len(mapr_list)):
        line = mapr_list[i].replace('\n', '').split(' ')
        k = int(line[0])
        v = line[1].strip().split(',')
        v = [int(e) for e in v]
        v = tuple(v)
        mapr_dict[k] = v

    # for testing
    test_size = 0
    test_f = open('dataset/test_list.txt', 'r')
    test_images, test_depths, test_labels = [], [], []
    for e in test_f.readlines():
        line = e.replace('\n', '').split()
        test_images.append(line[0])
        test_depths.append(line[1])
        test_labels.append(line[2])
        test_size += 1
    test_f.close()

    counter = 0
    time_counter = 0
    for n in range(test_size):
        # data forward
        image = cv2.imread(test_images[n], 1)
        depth = cv2.imread(test_depths[n], 0)

        time_start = time.time()
        segpred_numpy = model.forward(image, depth)
        time_counter += time.time() - time_start

        # data label to rgb
        if isTrans:
            segpred_w, segpred_h = segpred_numpy.shape
            segpred_rgb = np.zeros(shape=(segpred_w, segpred_h, 3), dtype=np.uint8)
            for i in range(segpred_w):
                for j in range(segpred_h):
                    segpred_rgb[i][j] = mapr_dict[segpred_numpy[i][j]]
            segpred_rgb = cv2.cvtColor(segpred_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(test_images[n].replace('dataset/rgb', 'results/testing'), np.hstack((image, segpred_rgb)))

        # print info
        counter += 1
        print('now dealing with %d: %s' %(counter,
                                          test_images[n].replace('dataset/test/rgb', 'results/testing')))
        if counter >= 100:
                break
    print('time using: %f' %(float(time_counter) / float(counter)))
    '''
    # data forward
    image = cv2.imread("/home/westwell/Desktop/Project/DCNN3forTest/dataset/left.png", 1)
    depth = cv2.imread("/home/westwell/Desktop/Project/DCNN3forTest/dataset/test.png", 0)

    depth = depth / 127. * 255
    segpred_numpy = model.forward(image, depth)

    # data label to rgb
    if isTrans:
        segpred_w, segpred_h = segpred_numpy.shape
        segpred_rgb = np.zeros(shape=(segpred_w, segpred_h, 3), dtype=np.uint8)
        for i in range(segpred_w):
            for j in range(segpred_h):
                segpred_rgb[i][j] = mapr_dict[segpred_numpy[i][j]]
        segpred_rgb = cv2.cvtColor(segpred_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite('results/testing/test.png', segpred_rgb)
    '''
    # for training
    train_size = 0
    train_f = open('dataset/train_list.txt', 'r')
    train_images, train_depths, train_labels = [], [], []
    for e in train_f.readlines():
        line = e.replace('\n', '').split(' ')
        train_images.append(line[0])
        train_depths.append(line[1])
        train_labels.append(line[2])
        train_size += 1
    train_f.close()

    counter = 0
    time_counter = 0
    for n in range(train_size):
        # data forward
        image = cv2.imread(train_images[n], 1)
        depth = cv2.imread(train_depths[n], 0)
        time_start = time.time()
        segpred_numpy = model.forward(image, depth)
        time_counter += time.time() - time_start

        # data label to rgb
        if isTrans:
            segpred_w, segpred_h = segpred_numpy.shape
            segpred_rgb = np.zeros(shape=(segpred_w, segpred_h, 3), dtype=np.uint8)
            for i in range(segpred_w):
                for j in range(segpred_h):
                    segpred_rgb[i][j] = mapr_dict[segpred_numpy[i][j]]
            segpred_rgb = cv2.cvtColor(segpred_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(train_images[n].replace('dataset/rgb', 'results/training'), segpred_rgb)

        # print info
        counter += 1
        print('now dealing with %d: %s' %(counter, train_images[n].replace('dataset/train/rgb', 'results/training')))
    print('time using: %f' %(float(time_counter) / float(counter)))

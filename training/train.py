# -*- coding: utf-8 -*-
"""
Editor: westwell
Version: 1.0
Time: 2019/06/14
"""

import os
import time
import torch
import numpy as np
from params import BaseOptions
from models.Deeplab import Deeplab_Solver
from data.custom_dataset_data_loader import CustomDatasetDataLoader
import utils.util as util

def main_task():

    # define params
    opt = BaseOptions().parse()
    iter_path = os.path.join(opt.checkpoints_dir, 'iter.txt')
    ioupath_path = os.path.join(opt.checkpoints_dir, 'MIoU.txt')

    # load training data
    if opt.continue_train:
        try:
            start_epoch, epoch_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_epoch, epoch_iter = 1, 0
        try:
            best_iou = np.loadtxt(ioupath_path, dtype=float)
        except:
            best_iou = 0.
    else:
        start_epoch, epoch_iter = 1, 0
        best_iou = 0.

    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_ids[0])

    # define data mode
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    dataset, dataset_val = data_loader.load_data()
    dataset_size = len(dataset)

    # define model
    model = Deeplab_Solver(opt)
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter

    print("starting training model......")

    for epoch in range(start_epoch, opt.nepochs):
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size

        # for train
        opt.isTrain = True
        model.model.train()
        for i, data in enumerate(dataset, start=epoch_iter):
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            # keep time to watch how times each one epoch
            epoch_start_time = time.time()

            # forward and backward pass
            model.forward(data, isTrain=True)
            model.backward(total_steps, opt.nepochs * dataset_size * opt.batchSize + 1)

            # save latest model
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if model.trainingavgloss < 0.010:
            break

        # for eval
        opt.isTrain = False
        model.model.eval()
        if dataset_val != None:
            label_trues, labels_preds = [], []
            for i, data in enumerate(dataset_val):
                seggt, segpred = model.forward(data, isTrain=False)
                seggt = seggt.data.cpu().numpy()
                segpred = segpred.data.cpu().numpy()

                label_trues.append(seggt)
                labels_preds.append(segpred)

            metrics = util.label_accuracy_score(
                    label_trues, labels_preds, n_class=opt.label_nc)
            metrics *= 100
            print('''\
                    Validation:
                    Accuracy: {0}
                    AccuracyClass: {1}
                    MeanIOU: {2}
                    FWAVAccuracy: {3}
                    '''.format(*metrics))

            # save model for best
            if metrics[2] > best_iou:
                best_iou = metrics[2]
                model.save('best')

            print('end of epoch %d / %d \t Time Taken: %d sec' % (epoch + 1, opt.nepochs, time.time() - epoch_start_time))

import sys
def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)
if __name__ == "__main__":
    try:
        main_task()
    except:
        print('start again')
        time.sleep(3)
        restart_program()

'''
if __name__ == "__main__":
    main_task()
'''


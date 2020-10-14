import cv2
import time
import numpy as np
from data.base_dataset import BaseDataset, get_params, transform


def make_dataset_fromlst(listfilename):
    images, depths, segs = [], [], []

    with open(listfilename) as f:
        content = f.readlines()
        for x in content:
            imgname, depthname, segname = x.strip().split(' ')
            images += [imgname]
            depths += [depthname]
            segs += [segname]

    return {'images':images, 'segs':segs, 'depths':depths}

class NYUDataset():
    def initialize(self, opt):
        self.opt = opt
        np.random.seed(int(time.time()))
        self.paths_dict = make_dataset_fromlst(opt.list)
        self.len = len(self.paths_dict['images'])

    def __getitem__(self, index):
        image = cv2.imread(self.paths_dict['images'][index], 1)
        depth = cv2.imread(self.paths_dict['depths'][index], 0).astype(np.float32)
        seg = cv2.imread(self.paths_dict['segs'][index], 0).astype(np.uint8)

        params = get_params(self.opt, seg.shape)
        depth_tensor_tranformed = transform(depth, params, normalize=False, istrain=self.opt.isTrain)
        seg_tensor_tranformed = transform(seg, params, normalize=False, method='nearest', istrain=self.opt.isTrain)

        if self.opt.inputmode == 'bgr-mean':
            img_tensor_tranformed = transform(image, params, normalize=False, istrain=self.opt.isTrain, option=1)
        else:
            img_tensor_tranformed = transform(image, params, istrain=self.opt.isTrain, option=1)

        return {'image':img_tensor_tranformed,
                'depth':depth_tensor_tranformed,
                'seg': seg_tensor_tranformed,
                'imgpath': self.paths_dict['segs'][index]}

    def __len__(self):
        return self.len

    def name(self):
        return 'NYUDataset_TRAIN'

class NYUDataset_val():
    def initialize(self, opt):
        self.opt = opt
        np.random.seed(8964)
        self.paths_dict = make_dataset_fromlst(opt.vallist)
        self.len = len(self.paths_dict['images'])

    def __getitem__(self, index):

        image = cv2.imread(self.paths_dict['images'][index], 1)
        depth = cv2.imread(self.paths_dict['depths'][index], 0).astype(np.float32)
        seg = cv2.imread(self.paths_dict['segs'][index], 0).astype(np.uint8)

        params = get_params(self.opt, seg.shape, test=True)
        depth_tensor_tranformed = transform(depth, params, normalize=False, istrain=self.opt.isTrain)
        seg_tensor_tranformed = transform(seg, params, normalize=False, method='nearest', istrain=self.opt.isTrain)

        if self.opt.inputmode == 'bgr-mean':
            img_tensor_tranformed = transform(image, params, normalize=False, istrain=self.opt.isTrain, option=1)
        else:
            img_tensor_tranformed = transform(image, params, istrain=self.opt.isTrain, option=1)

        return {'image':img_tensor_tranformed,
                'depth':depth_tensor_tranformed,
                'seg': seg_tensor_tranformed,
                'imgpath': self.paths_dict['segs'][index]}

    def __len__(self):
        return self.len

    def name(self):
        return 'NYUDataset_VALID'

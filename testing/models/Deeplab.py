import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

import models.VGG_Deeplab as VGG_Deeplab


class Deeplab_VGG(nn.Module):
    def __init__(self, num_classes, depthconv, is_train):
        super(Deeplab_VGG, self).__init__()
        self.Scale = VGG_Deeplab.vgg16_bn(num_classes=num_classes, depthconv=depthconv, is_train=is_train)
        for param in self.Scale.parameters():
            param.requires_grad = True

    def forward(self, x, depth=None):
        output = self.Scale(x, depth)
        return output

class Deeplab_Solver():
    def __init__(self, opt):
        self.opt = opt
        self.model = Deeplab_VGG(self.opt.label_nc, self.opt.depthconv, self.opt.isTrain)

        if self.opt.isTrain:
            self.criterionSeg = torch.nn.CrossEntropyLoss(ignore_index=255).cuda()

            self.optimizer = torch.optim.SGD([{'params': self.model.Scale.get_1x_lr_params_NOscale(), 'lr': self.opt.lr},
                                              {'params': self.model.Scale.get_10x_lr_params(), 'lr': self.opt.lr},
                                              {'params': self.model.Scale.get_2x_lr_params_NOscale(), 'lr': self.opt.lr, 'weight_decay': 0.},
                                              {'params': self.model.Scale.get_20x_lr_params(), 'lr': self.opt.lr, 'weight_decay': 0.}],
                                              lr=self.opt.lr, momentum=self.opt.momentum, weight_decay=self.opt.wd)
            self.averageloss = []

        if not self.opt.isTrain or self.opt.continue_train:
            self.load()
        if len(self.opt.gpu_ids) and torch.cuda.is_available():
            self.model.cuda()

    def forward(self, data, isTrain=True):
        self.model.zero_grad()

        self.image = Variable(data['image'], volatile=not isTrain).cuda()
        self.depth = Variable(data['depth'], volatile=not isTrain).cuda()
        if data['seg'] is not None:
            self.seggt = Variable(data['seg'], requires_grad=isTrain, volatile=not isTrain).cuda()
        else:
            self.seggt = None

        input_size = self.image.size()

        self.segpred = self.model(self.image, self.depth)
        self.segpred = nn.functional.upsample(self.segpred, size=(input_size[2], input_size[3]), mode='bilinear')
        self.segpred_max = self.segpred.max(1, keepdim=True)[1]

        if self.opt.isTrain:
            self.loss = self.criterionSeg(self.segpred, torch.squeeze(self.seggt, 1).long())
            self.averageloss += [self.loss.cpu().detach().numpy()]

        return self.seggt, self.segpred_max

    def backward(self, step, total_step):

        self.loss.backward()
        self.optimizer.step()
        if step % self.opt.iterSize  == 0:
            self.current_lr = self.update_learning_rate(step, total_step)

        if step % self.opt.print_freq == 0:
            self.trainingavgloss = np.mean(self.averageloss)
            print('freq loss: %f, learning rate: %f' %(self.trainingavgloss, self.current_lr))
            self.averageloss = []

    def save(self, which_epoch):
        save_filename = '%s_net_%s.pt' % (which_epoch, 'net')
        save_path = os.path.join(self.opt.checkpoints_dir, save_filename)
        torch.save(self.model.cpu().state_dict(), save_path)
        if len(self.opt.gpu_ids) and torch.cuda.is_available():
            self.model.cuda()

    def load(self):
        self.model.load_state_dict(torch.load('./checkpoints/latest_net_net.pt'))

    def update_learning_rate(self, step, total_step):

        lr = max(self.opt.lr * ((1 - float(step) / total_step) ** (self.opt.lr_power)), 1e-6)

        self.optimizer.param_groups[0]['lr'] = lr
        self.optimizer.param_groups[1]['lr'] = lr
        self.optimizer.param_groups[2]['lr'] = lr
        self.optimizer.param_groups[3]['lr'] = lr

        return lr

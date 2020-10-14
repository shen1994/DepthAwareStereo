import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.model_utils import *
from .ops.depthconv.modules import DepthConv
from .ops.depthavgpooling.modules import Depthavgpooling

class ConvModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1,
                 bn=False, maxpool=False, pool_kernel=3, pool_stride=2, pool_pad=1):
        super(ConvModule, self).__init__()
        conv2d = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride,
                           padding=padding, dilation=dilation)
        layers = []
        if bn:
            layers += [nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
        else:
            layers += [nn.ReLU(inplace=True)]
        if maxpool:
            layers += [nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_pad)]
        self.layers = nn.Sequential(*([conv2d]+layers))

    def forward(self, x):
        x = self.layers(x)
        return x

class DepthConvModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1,bn=False):
        super(DepthConvModule, self).__init__()

        conv2d = DepthConv(inplanes,planes, kernel_size=kernel_size, stride=stride,
                           padding=padding, dilation=dilation)
        layers = []
        if bn:
            layers += [nn.BatchNorm2d(planes), nn.ReLU(inplace=True)]
        else:
            layers += [nn.ReLU(inplace=True)]
        self.layers = nn.Sequential(*([conv2d]+layers))

    def forward(self, x, depth):

        for im,module in enumerate(self.layers._modules.values()):
            if im==0:
                x = module(x,depth)
            else:
                x = module(x)
        return x

class VGG_layer(nn.Module):

    def __init__(self, batch_norm=False, depthconv=False):
        super(VGG_layer, self).__init__()

        self.depthconv = depthconv
        if self.depthconv:
            self.conv1_1_depthconvweight = 1.
            self.conv1_1 = DepthConvModule(3, 64, bn=batch_norm)
        else:
            self.conv1_1 = ConvModule(3, 64, bn=batch_norm)
        self.conv1_2 = ConvModule(64, 64, bn=batch_norm, maxpool=True)

        if self.depthconv:
            self.conv2_1_depthconvweight = 1.
            self.downsample_depth2_1 = nn.AvgPool2d(3,padding=1,stride=2)
            self.conv2_1 = DepthConvModule(64, 128, bn=batch_norm)
        else:
            self.conv2_1 = ConvModule(64, 128, bn=batch_norm)
        self.conv2_2 = ConvModule(128, 128, bn=batch_norm, maxpool=True)

        if self.depthconv:
            self.conv3_1_depthconvweight = 1.
            self.downsample_depth3_1 = nn.AvgPool2d(3,padding=1,stride=2)
            self.conv3_1 = DepthConvModule(128, 256, bn=batch_norm)
        else:
            self.conv3_1 = ConvModule(128, 256, bn=batch_norm)
        self.conv3_2 = ConvModule(256, 256, bn=batch_norm)
        self.conv3_3 = ConvModule(256, 256, bn=batch_norm, maxpool=True)

        if self.depthconv:
            self.conv4_1_depthconvweight = 1.
            self.downsample_depth4_1 = nn.AvgPool2d(3,padding=1,stride=2)
            self.conv4_1 = DepthConvModule(256, 512, bn=batch_norm)
        else:
            self.conv4_1 = ConvModule(256, 512, bn=batch_norm)
        self.conv4_2 = ConvModule(512, 512, bn=batch_norm)
        self.conv4_3 = ConvModule(512, 512, bn=batch_norm,
                                  maxpool=True, pool_kernel=3, pool_stride=1, pool_pad=1)

        if self.depthconv:
            self.conv5_1_depthconvweight = 1.#nn.Parameter(torch.ones(1))
            self.conv5_1 = DepthConvModule(512, 512, bn=batch_norm,dilation=2,padding=2)
        else:
            self.conv5_1 = ConvModule(512, 512, bn=batch_norm, dilation=2, padding=2)
        self.conv5_2 = ConvModule(512, 512, bn=batch_norm, dilation=2, padding=2)
        self.conv5_3 = ConvModule(512, 512, bn=batch_norm, dilation=2, padding=2,
                                  maxpool=True, pool_kernel=3, pool_stride=1, pool_pad=1)
        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1,padding=1)
        self.pool5a_d = Depthavgpooling(kernel_size=3, stride=1,padding=1)

    def forward(self, x, depth=None):
        if self.depthconv:
            x = self.conv1_1(x,self.conv1_1_depthconvweight * depth)
        else:
            x = self.conv1_1(x)
        x = self.conv1_2(x)
        if self.depthconv:
            depth = self.downsample_depth2_1(depth)
            x = self.conv2_1(x, self.conv2_1_depthconvweight * depth)
        else:
            x = self.conv2_1(x)

        x = self.conv2_2(x)
        if self.depthconv:
            depth = self.downsample_depth3_1(depth)
            x = self.conv3_1(x, self.conv3_1_depthconvweight * depth)
        else:
            x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        if self.depthconv:
            depth = self.downsample_depth4_1(depth)
            x = self.conv4_1(x, self.conv4_1_depthconvweight * depth)
        else:
            x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        if self.depthconv:
            x = self.conv5_1(x, self.conv5_1_depthconvweight * depth)
        else:
            x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        if self.depthconv:
            x = self.pool5a_d(x,depth)
        else:
            x = self.pool5a(x)

        return x, depth

class Classifier_Module(nn.Module):

    def __init__(self, num_classes, inplanes, is_train, depthconv=False):
        super(Classifier_Module, self).__init__()
        self.depthconv = depthconv
        self.is_train = is_train
        if depthconv:
            self.fc6_2_depthconvweight = 1.
            self.fc6_2 = DepthConv(inplanes, 1024, kernel_size=3, stride=1, padding=12, dilation=12)
            self.downsample_depth = None
        else:
            self.downsample_depth = nn.AvgPool2d(9,padding=1,stride=8)
            self.fc6_2 = nn.Conv2d(inplanes, 1024, kernel_size=3, stride=1, padding=12, dilation=12)  # fc6

        self.fc7_2 = nn.Sequential(
            *[nn.ReLU(True), nn.Dropout(),
              nn.Conv2d(1024, 1024, kernel_size=1, stride=1), nn.ReLU(True), nn.Dropout()])  # fc7

        self.globalpooling = nn.AdaptiveAvgPool2d((1, 1))
        if self.is_train:
            self.dropout = nn.Dropout(0.3)
        self.fc8_2 = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, bias=True)  # fc8

    def forward(self, x, depth=None):
        if self.depthconv:
            out2 = self.fc6_2(x, self.fc6_2_depthconvweight * depth)
        else:
            out2 = self.fc6_2(x)
        out2 = self.fc7_2(out2)
        out2_size = out2.size()

        globalpool = self.globalpooling(out2)
        if self.is_train:
            globalpool = self.dropout(globalpool)
        upsample = nn.Upsample((out2_size[2],out2_size[3]), mode='bilinear')
        globalpool = upsample(globalpool)

        out2 = torch.cat([out2, globalpool], 1)
        out2 = self.fc8_2(out2)
        return out2

class VGG(nn.Module):

    def __init__(self, num_classes=20, init_weights=True, depthconv=False, bn=False, is_train=True):
        super(VGG, self).__init__()
        self.features = VGG_layer(batch_norm=bn,depthconv=depthconv)
        self.classifier = Classifier_Module(num_classes, 512, depthconv=depthconv, is_train=is_train)

        if init_weights:
            self._initialize_weights()

    def forward(self, x, depth=None):
        x,depth = self.features(x,depth)
        x = self.classifier(x,depth)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_normalize_params(self):
        b=[]
        b.append(self.classifier.norm)
        for i in b:
            if isinstance(i, CaffeNormalize):
                yield i.scale

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.features.conv1_1)
        b.append(self.features.conv1_2)
        b.append(self.features.conv2_1)
        b.append(self.features.conv2_2)
        b.append(self.features.conv3_1)
        b.append(self.features.conv3_2)
        b.append(self.features.conv3_3)
        b.append(self.features.conv4_1)
        b.append(self.features.conv4_2)
        b.append(self.features.conv4_3)
        b.append(self.features.conv5_1)
        b.append(self.features.conv5_2)
        b.append(self.features.conv5_3)
        b.append(self.classifier.fc6_2)
        b.append(self.classifier.fc7_2)

        for i in range(len(b)):
            for j in b[i].modules():
                if isinstance(j, nn.Conv2d):
                    if j.weight.requires_grad:
                        yield j.weight
                elif isinstance(j, DepthConv):
                    if j.weight.requires_grad:
                        yield j.weight

    def get_2x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.features.conv1_1)
        b.append(self.features.conv1_2)
        b.append(self.features.conv2_1)
        b.append(self.features.conv2_2)
        b.append(self.features.conv3_1)
        b.append(self.features.conv3_2)
        b.append(self.features.conv3_3)
        b.append(self.features.conv4_1)
        b.append(self.features.conv4_2)
        b.append(self.features.conv4_3)
        b.append(self.features.conv5_1)
        b.append(self.features.conv5_2)
        b.append(self.features.conv5_3)
        b.append(self.classifier.fc6_2)
        b.append(self.classifier.fc7_2)

        for i in range(len(b)):
            for j in b[i].modules():
                if isinstance(j, nn.Conv2d):
                    if j.bias is not None:
                        if j.bias.requires_grad:
                            yield j.bias
                elif isinstance(j, DepthConv):
                    if j.bias is not None:
                        if j.bias.requires_grad:
                            yield j.bias

    def get_10x_lr_params(self):
        b = []
        b.append(self.classifier.fc8_2.weight)

        for i in b:
            yield i

    def get_20x_lr_params(self):
        b = []
        b.append(self.classifier.fc8_2.bias)

        for i in b:
            yield i

    def get_100x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.features.conv1_1_depthconvweight)
        b.append(self.features.conv2_1_depthconvweight)
        b.append(self.features.conv3_1_depthconvweight)
        b.append(self.features.conv4_1_depthconvweight)
        b.append(self.features.conv5_1_depthconvweight)
        b.append(self.classifier.fc6_1_depthconvweight)
        b.append(self.classifier.fc6_2_depthconvweight)
        b.append(self.classifier.fc6_3_depthconvweight)
        b.append(self.classifier.fc6_4_depthconvweight)

        for j in range(len(b)):
            yield b[j]

def vgg16(**kwargs):
    model = VGG(bn=False,**kwargs)
    return model

def vgg16_bn(**kwargs):
    model = VGG(bn=True, **kwargs)
    return model

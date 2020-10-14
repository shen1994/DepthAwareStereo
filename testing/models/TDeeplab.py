import torch
import numpy as np
from torch.autograd import Variable
import models.VGG_Deeplab as VGG_Deeplab

class Deeplab_VGG(torch.nn.Module):
    def __init__(self, num_classes, depthconv, is_train):
        super(Deeplab_VGG, self).__init__()
        self.Scale = VGG_Deeplab.vgg16(num_classes=num_classes, depthconv=depthconv, is_train=is_train)

    def forward(self,x, depth=None):
        output = self.Scale(x, depth)
        return output

class Deeplab_Solver():
    def __init__(self, label_nc=2, model_path=''):
        super(Deeplab_Solver, self).__init__()
        self.model = Deeplab_VGG(label_nc, True, False)
        self.model.load_state_dict(torch.load(model_path))
        self.model.cuda()

    def forward(self, _image, _depth):

        _image = _image - np.asarray([122.675, 116.669, 104.008])
        _image = _image.transpose((2, 0, 1))[::-1, :, :].astype(np.float32)
        _depth = _depth.astype(np.float32)

        image_tensor = torch.from_numpy(_image).float().unsqueeze(0)
        depth_tensor = torch.from_numpy(np.expand_dims(_depth, axis=0)).float().unsqueeze(0)

        with torch.no_grad():        
            image = Variable(image_tensor).cuda()
            depth = Variable(depth_tensor).cuda()

            input_size = image.size()
            
            segpred = self.model(image, depth)
            segpred = torch.nn.functional.upsample(segpred, size=(input_size[2], input_size[3]),  mode='bilinear', align_corners=True)
            segpred = segpred.max(1, keepdim=True)[1]
            segpred_numpy = segpred.data.cpu().numpy()[0][0]

        return segpred_numpy

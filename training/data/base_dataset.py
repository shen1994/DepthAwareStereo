import cv2
import torch
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size, test=False):
    h, w = size
    if opt.scale and test==False:
        scale = random.uniform(1., 1.25)
        new_h = h * scale
        new_w = (new_h * w // h)

        new_h = int(round(new_h / 8) * 8)
        new_w = int(round(new_w / 8) * 8)
    else:
        new_h = h
        new_w = w

    if opt.flip and test==False:
        flip = random.random() > 0.5
    else:
        flip = False

    if opt.noise and test==False:
        noise = random.random() > 0.5
    else:
        noise = False

    if opt.colorjitter and test==False:
        colorjitter = True
    else:
        colorjitter = False
    return {'scale': (new_w, new_h),
            'flip': flip,
            'noise': noise,
            'colorjitter': colorjitter}

def center_scale_transform(numpyarray, target_shape, method):
    nshape = numpyarray.shape
    nshapew = nshape[0]
    nshapeh = nshape[1]
    if method == 'linear':
        numpyarray_temp = cv2.resize(numpyarray, (target_shape[0], target_shape[1]), interpolation=cv2.INTER_LINEAR).copy()
    else:
        numpyarray_temp = cv2.resize(numpyarray, (target_shape[0], target_shape[1]), interpolation=cv2.INTER_NEAREST).copy()

    if target_shape[1] > nshapew:
        upw_start = 0
        upw_end = nshapew
        downw_start = (target_shape[1] - nshapew) // 2
        downw_end = downw_start + nshapew
    else:
        upw_start = (nshapew - target_shape[1]) // 2
        upw_end = upw_start + target_shape[1]
        downw_start = 0
        downw_end = target_shape[1]

    if target_shape[0] > nshapeh:
        uph_start = 0
        uph_end = nshapeh
        downh_start = (target_shape[0] - nshapeh) // 2
        downh_end = downh_start + nshapeh
    else:
        uph_start = (nshapeh - target_shape[0]) // 2
        uph_end = uph_start + target_shape[0]
        downh_start = 0
        downh_end = target_shape[0]

    numpyarray = np.zeros_like(numpyarray)
    numpyarray[upw_start:upw_end, uph_start:uph_end, ...] = \
        numpyarray_temp[downw_start:downw_end, downh_start:downh_end, ...]

    return numpyarray

def random_scale_transform(numpyarray, target_shape, method):
    nshape = numpyarray.shape
    nshapew = nshape[0]
    nshapeh = nshape[1]
    if method == 'linear':
        numpyarray_temp = cv2.resize(numpyarray, (target_shape[0], target_shape[1]), interpolation=cv2.INTER_LINEAR).copy()
    else:
        numpyarray_temp = cv2.resize(numpyarray, (target_shape[0], target_shape[1]), interpolation=cv2.INTER_NEAREST).copy()

    if target_shape[1] > nshapew:
        upw_start = 0
        upw_end = nshapew
        downw_start = np.random.randint(0, target_shape[1] - nshapew)
        downw_end = downw_start + nshapew
    elif target_shape[1] < nshapew:
        upw_start = np.random.randint(0, nshapew - target_shape[1])
        upw_end = upw_start + target_shape[1]
        downw_start = 0
        downw_end = target_shape[1]
    else:
        upw_start = 0
        upw_end = nshapew
        downw_start = 0
        downw_end = nshapew

    if target_shape[0] > nshapeh:
        uph_start = 0
        uph_end = nshapeh
        downh_start = np.random.randint(0, target_shape[0] - nshapeh)
        downh_end = downh_start + nshapeh
    elif target_shape[0] < nshapeh:
        uph_start = np.random.randint(0, nshapeh - target_shape[0])
        uph_end = uph_start + target_shape[0]
        downh_start = 0
        downh_end = target_shape[0]
    else:
        uph_start = 0
        uph_end = nshapeh
        downh_start = 0
        downh_end = nshapeh

    numpyarray = np.zeros_like(numpyarray)
    numpyarray[upw_start:upw_end, uph_start:uph_end, ...] = \
        numpyarray_temp[downw_start:downw_end, downh_start:downh_end, ...]

    return numpyarray

def gaussian_noise(im, mean=0.2, sigma=0.5):
    for i in range(len(im)):
        im[i] += random.gauss(mean, sigma)
    return im

def image_gaussian_noise(image):
    w, h = image.shape[:2]
    image_r = gaussian_noise(image[:,:,0].flatten())
    image_g = gaussian_noise(image[:,:,1].flatten())
    image_b = gaussian_noise(image[:,:,2].flatten())
    image[:,:,0] = image_r.reshape([w, h])
    image[:,:,1] = image_g.reshape([w, h])
    image[:,:,2] = image_b.reshape([w, h])
    return image.astype(np.uint8)

def transform(numpyarray, params, normalize=True, method='linear', istrain=True, colorjitter=False, option=0):
    if istrain:

        # innner window is moving not at the center
        # numpyarray = center_scale_transform(numpyarray, params['scale'], method)
        # random moving the inner window, cause different sight
        numpyarray = random_scale_transform(numpyarray, params['scale'], method)

        if params['flip']:
            numpyarray = cv2.flip(numpyarray, 1)

        if option==1:
            if params['noise']:
                numpyarray = image_gaussian_noise(numpyarray)

            if colorjitter and params['colorjitter'] and random.random() > 0.1:
                numpyarray = Image.fromarray(np.uint8(numpyarray))
                random_factor = np.random.randint(0, 31) / 10.
                color_image = ImageEnhance.Color(numpyarray).enhance(random_factor)
                random_factor = np.random.randint(10, 21) / 10.
                brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
                random_factor = np.random.randint(10, 21) / 10.
                contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
                random_factor = np.random.randint(0, 31) / 10.
                numpyarray = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)

    if option == 1:
        if not normalize:
            numpyarray = numpyarray - np.asarray([122.675,116.669,104.008])
            numpyarray = numpyarray.transpose((2, 0, 1))[::-1,:,:].astype(np.float32)
        else:
            numpyarray = numpyarray.transpose((2, 0, 1)).astype(np.float32)/255.

    if option == 2:
        if not normalize:
            numpyarray = numpyarray - np.asarray([132.431, 94.076, 118.477])
            numpyarray = numpyarray.transpose((2, 0, 1))[::-1,:,:].astype(np.float32)
        else:
            numpyarray = numpyarray.transpose((2, 0, 1)).astype(np.float32)/255.

    if len(numpyarray.shape) == 3:
        torchtensor = torch.from_numpy(numpyarray).float()#.div(255)
    else:
        torchtensor = torch.from_numpy(np.expand_dims(numpyarray,axis=0))

    if normalize:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        torchtensor = normalize(torchtensor)

    return torchtensor

## Basic Python imports
import math
import random 
from io import BytesIO

## PyTorch imports
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF 

## Image / Array imports
from PIL import Image
from PIL import ImageFilter
import numpy as np 


import matplotlib as mpl 
import matplotlib.pyplot as plt 

class RotationTransform(object):
    """
    Rotates a PIL image by 0, 90, 180, or 270 degrees. Randomly chosen using a uniform distribution.
    """
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, image):
        angle = random.choice(self.angles)
        return TF.rotate(image, angle)

class GammaJitter(object):
    """
    Jitters the gamma of a PIL image between a uniform distribution of two values (low & high).
    Larger gammas make the shadows darker, smaller gammas make the shadows lighter.
    """
    def __init__(self, low=0.9, high=1.1):
        self.low = low
        self.high = high
    
    def __call__(self, image):
        gamma = np.random.uniform(self.low, self.high)
        return TF.adjust_gamma(image, gamma)

class BrightnessJitter(object):
    """
    Jitters the gamma of a PIL image between a uniform distribution of two values (low & high).
    Larger gammas make the shadows darker, smaller gammas make the shadows lighter.
    """
    def __init__(self, low=0.9, high=1.1):
        self.low = low
        self.high = high
    
    def __call__(self, image):
        factor = np.random.uniform(self.low, self.high)
        return TF.adjust_brightness(image, factor)

class RandomScale(object):
    """
    Scales a PIL image based on a value chosen from a uniform distribution of two values (low & high).
    """
    def __init__(self, low=1.0, high=1.1):
        self.low = low
        self.high = high

    def __call__(self, image):
        height = image.height
        width = image.width
        scale = np.random.uniform(self.low, self.high)
        image = TF.resize(image, (math.floor(height * scale), math.floor(width * scale)))
        return TF.center_crop(image, (height, width))

class Interpolate(object):
    """
    Resizes an image to a certain (square) size or by a scale factor.
    """
    def __init__(self, scale=1, size=0):
        self.scale = scale
        self.size = size

    def __call__(self, image):
        if self.size == 0:
            height = image.height
            width = image.width
            return image.resize(((width * self.scale) // 1, (height * self.scale) // 1), Image.BILINEAR)
        else:
            return image.resize((self.size, self.size), Image.BILINEAR)

class InterpolateTorch(object):
    """
    Resizes an image to a certain (square) size or by a scale factor.
    """
    def __init__(self, scale=1, size=0):
        self.scale = scale
        self.size = size

    def __call__(self, image):
        if self.size == 0:
            height = image.shape[1]
            width = image.shape[2]
            if len(image.shape) == 3:
                image = torch.unsqueeze(image, 0)
            return F.interpolate(image, scale_factor=self.scale, mode='bilinear', align_corners=True)
        else:
            return F.interpolate(image, size=(self.size, self.size), mode='bilinear', align_corners=True)

class MaxPool(object):
    """
    Performs Max Pooling on a PIL Image.
    """
    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.to_tensor = TF.to_tensor
        self.to_pil = TF.to_pil_image

    def __call__(self, image):
        image = self.to_tensor(image)
        image = F.max_pool2d(image, self.kernel_size, self.stride)
        image = self.to_pil(image)
        return image

class MaxPoolTorch(object):
    """
    Performs Max Pooling on a PyTorch tensor.
    """
    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size
        self.stride = kernel_size

    def __call__(self, image):
        image = F.max_pool2d(image, self.kernel_size, self.stride)
        return image

class AdaptiveCenterCrop(object):
    """
    Center crops the image to be a square of the smallest edge squared.
    """
    def __init__(self):
        pass

    def __call__(self, image):
        length = min(image.width, image.height)
        return TF.center_crop(image, (length, length))

class MedianFilter(object):
    """
    Randomly applies a median filter to an image.
    """
    def __init__(self, filter_size=3, p=0.1):
        self.filter_size = filter_size
        self.p = p

    def __call__(self, image):
        roll = random.random()
        if roll < self.p:
            return image.filter(ImageFilter.MedianFilter(self.filter_size))
        else:
            return image

class Squeeze(object):
    """
    Squeezes the given PyTorch Tensor object
    """
    def __init__(self):
        ...

    def __call__(self, image):
        if len(image.shape) == 4:
            return image.squeeze_(0)
        else:
            return image

class Unsqueeze(object):
    """
    Unsqueezes the given PyTorch Tensor object in axis=0
    """
    def __init__(self):
        ...

    def __call__(self, image):
        if len(image.shape) == 3:
            return image.unsqueeze_(0)
        else:
            return image

class JPEGCompression(object):
    """
    Compresses the image using jpeg compression and restores
    """
    def __init__(self, quality=75):
        self.quality = quality
    
    def __call__(self, image):
        save = BytesIO()
        image.save(save, 'JPEG', quality=self.quality)
        image = Image.open(save)
        return image

class JPEGCompressionTorch(object):
    """
    Compresses the image using jpeg compression and restores
    """
    def __init__(self, quality=75):
        self.quality = quality
    
    def __call__(self, image):
        assert torch.is_tensor(image)
        image = transforms.ToPILImage()(image.cpu().squeeze())
        save = BytesIO()
        image.save(save, 'JPEG', quality=self.quality)
        image = transforms.ToTensor()(Image.open(save))
        return (image.unsqueeze(0).cuda()).requires_grad_(True)

class RandomResizeAndPadTorch(object):
    """
    Randomly resizes and pads an image.
    """
    def __init__(self, initial_size, final_size):
        self.initial_size = initial_size
        self.final_size = final_size
        self.resize = F.interpolate
        self.pad = F.pad

    def __call__(self, image):
        # image = image.unsqueeze(0)

        # Resize the image
        resize_shape = random.randint(self.initial_size, self.final_size)
        image = self.resize(image, size=(resize_shape, resize_shape), mode='bilinear', align_corners=True)

        # Determine how much to pad and do it
        remainder = self.final_size - resize_shape
        left = random.randint(0, remainder)
        right = remainder - left
        top = random.randint(0, remainder)
        bottom = remainder - top
        image = self.pad(image, pad=(left, right, top, bottom), mode='constant', value=0)

        return image

class Normalize(object):
    """
    Normalizes a batch of RGB images [n, 3, H, W]
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, image):
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        image[:, 0, :, :] = (image[:, 0, :, :] - self.mean[0]) / self.std[0]
        image[:, 1, :, :] = (image[:, 1, :, :] - self.mean[1]) / self.std[1]
        image[:, 2, :, :] = (image[:, 2, :, :] - self.mean[2]) / self.std[2]
        image = image.squeeze()
        return image
       
## Basic Python imports
import math
import random 

## PyTorch imports
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF 

## Image / Array imports
from PIL import Image
import numpy as np 

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

class RandomScale(object):
    """
    Scales a PIL image based on a value chosen from a uniform distribution of two values (low & high).
    """
    def __init__(self, low=1.0, high=1.1):
        self.low = low
        self.high = high

    def __call__(self, image):
        scale = np.random.uniform(self.low, self.high)
        image = TF.resize(image, (math.floor(image.height * scale), math.floor(image.width * scale)))
        return TF.center_crop(image, (256, 256))
        
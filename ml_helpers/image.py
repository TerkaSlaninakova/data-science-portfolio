from .imports import *
import os
from scipy import ndimage

def load_image_from_path(path):
    if not os.path.exists(path):
        return None
    else:
        return ndimage.imread(path)
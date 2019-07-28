from .imports import *
from urllib.request import urlretrieve
import os
import gzip
import pickle

def download_dataset(url, path):
    if not os.path.exists(path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        urlretrieve(url, path)

def open_gzip(filename):
    return gzip.open(filename, 'rb')

def load_pickle(dataset):
    """
    Loads a pickle dataset. Returns a tuple:
    Example (MNIST from http://deeplearning.net/data/mnist/mnist.pkl.gz):
    ((x, y), (x_valid, y_valid), _) = load_pickle(open_gzip(path+FILENAME))
    """
    return pickle.load(dataset, encoding='latin-1')
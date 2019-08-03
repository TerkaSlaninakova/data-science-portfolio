from .imports import *
from urllib.request import urlretrieve
import os
import gzip
import zipfile
import pickle
import shutil
import re

def download_dataset(url, path):
    if not os.path.exists(path):
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        urlretrieve(url, path)

def move_files_to_dir(dirname, path_to_files):
    dir_existed = False
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    else:
        dir_existed = True
    for f in path_to_files:
        if not dir_existed and not os.path.exists(dirname + f):
            shutil.move(f, dirname)

def find_files_in_dir(dir, regex=None):
    """
    Finds files in a directory, filters them with a regex (optional)
    Example: cat_filenames = find_files_in_dir(PATH + "\\train", "cat")
    """
    all_files = os.listdir(path=dir)
    if regex:
        found_files = [f for f in all_files if len(re.findall(regex, f)) != 0]
    else:
        found_files = all_files
    return found_files

def open_gzip(filename):
    return gzip.open(filename, 'rb')

def open_zip(filename):
    zip_ref = zipfile.ZipFile(filename, 'r')
    if len(zip_ref.infolist()) == 1:
        zip_ref.extractall(os.path.dirname(filename.split(".zip")[0]))
    else:
        zip_ref.extractall(filename.split(".zip")[0])
    zip_ref.close()
    return filename.split(".zip")[0]

def load_pickle(dataset):
    """
    Loads a pickle dataset. Returns a tuple:
    Example (MNIST from http://deeplearning.net/data/mnist/mnist.pkl.gz):
    ((x, y), (x_valid, y_valid), _) = load_pickle(open_gzip(path+FILENAME))
    """
    return pickle.load(dataset, encoding='latin-1')
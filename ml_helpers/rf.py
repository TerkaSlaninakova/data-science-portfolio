from .imports import *
from sklearn.ensemble import forest

# from https://github.com/fastai/fastai/blob/72286b8b22284a53b07d777783ff5d392a3d45b0/old/fastai/structured.py
def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))

# from https://github.com/fastai/fastai/blob/72286b8b22284a53b07d777783ff5d392a3d45b0/old/fastai/structured.py
def reset_rf_samples():
    """ Undoes the changes produced by set_rf_samples.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n_samples))
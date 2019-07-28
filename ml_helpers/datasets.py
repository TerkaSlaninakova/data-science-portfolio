from .imports import *
import mglearn
from sklearn import datasets

def get_two_class_classification_ds():
    """
    X,y = get_two_class_classification_ds()
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    """
    return mglearn.make_forge()

def get_two_class_regression_ds():
    """
    X,y = get_two_class_regression_ds()
    plt.plot(X, y, 'o')
    """
    return mglearn.make_wave()

def get_linear_relationship_with_randomness():
    """
    X,y = get_linear_relationship_with_randomness()
    plt.scatter(X, y)

    Note: Used to demonstrate uselessness of random forests in time dependent data
    (cannot extrapolate on validation set)
    y_trn,y_val = y[:40],y[40:]
    m = RandomForestRegressor().fit(x_trn,y_trn)
    m.predict(x_val) - won't return a linear relationship as expected
    """
    x = np.linspace(0,1)
    y = x + np.random.uniform(-0.2, 0.2, x.shape)
    return x,y

def get_non_linearly_separable_ds(centers=4):
    """
    X,y = get_non_linearly_separable_ds()
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    """
    return datasets.make_blobs(centers=centers)

def get_linearly_separable_ds():
    X, y = datasets.make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return (X, y)
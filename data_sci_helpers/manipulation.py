import numpy as np

def shuffle_data(dataset):
    """Shuffles the dataset, useful when creating training-validation batches
        if the data differs in order (e.g. when selecting last 20% for
        validation, the data would have different distribution.)
    Args:
    dataset: A `pd.DataFrame`, the dataset to shuffle.
    Returns:
    Shuffled dataset
    """
    return dataset.reindex(np.random.permutation(dataset.index))

def how_many_values_above_threshold(dataset, threshold, features = []):
    """Checks out how many data points are there above a given threshold.
        Useful in outlier detection.
    Args:
    dataset: A `pd.DataFrame`, the dataset
    threshold: A `number/int` threshold value above which the data points should be searched
    features: A `list` of features to check in the dataset
    Returns:
    A `dict` of feature: number of data points above threshold
    """
    feature_value_dict = {}
    for feature in features:
        feature_value_dict[feature] = dataset[dataset[feature] > threshold][feature].count()
    return feature_value_dict

def how_many_values_below_threshold(dataset, threshold, features = []):
    """Checks out how many data points are there below a given threshold.
        Useful in outlier detection.
    Args:
    dataset: A `pd.DataFrame`, the dataset
    threshold: Numerical threshold value below which the data points should be searched
    features: A `list` of features to check in the dataset
    Returns:
    A `dict` of feature: number of data points below threshold
    """
    feature_value_dict = {}
    for feature in features:
        feature_value_dict[feature] = dataset[dataset[feature] < threshold][feature].count()
    return feature_value_dict
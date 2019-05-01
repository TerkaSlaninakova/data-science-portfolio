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

def how_many_values_above_threshold(dataset, feature_name, threshold):
    """Checks out how many data points are there above a given threshold.
        Useful in outlier detection.
    Args:
    dataset: A `pd.DataFrame`, the dataset
    feature_name: A `string`, name of the feature to check
    threshold: A `number/int` threshold value above which the data points should be searched
    Returns:
    A `dict` of feature: number of data points above threshold
    """
    return dataset[dataset[feature_name] > threshold][feature_name].count()

def how_many_values_below_threshold(dataset, feature_name, threshold):
    """Checks out how many data points are there below a given threshold.
        Useful in outlier detection.
    Args:
    dataset: A `pd.DataFrame`, the dataset
    feature_name: A `string`, name of the feature to check
    threshold: Numerical threshold value below which the data points should be searched
    Returns:
    A `number` of times a given value was present within the feature
    """
    return dataset[dataset[feature_name] < threshold][feature_name].count()

def clip_values_above_threshold(feature, threshold):
    return feature.apply(lambda x: min(x, threshold))

def clip_values_below_threshold(feature, threshold):
    return feature.apply(lambda x: max(x, threshold))

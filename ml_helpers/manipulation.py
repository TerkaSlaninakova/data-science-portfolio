from .imports import *

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


def split_dataset(a, n): 
	"""
	Splits a dataset into 2 parts
	Usage:
	valid_size = 12000
	train_size = len(df_raw) - valid_size
	X_train, X_valid = split_dataset(df_raw, train_size)
	y_train, y_valid = split_dataset(y, train_size)
	"""
	return a[:n].copy(), a[n:].copy()

def get_random_df_sample(df, n):
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()
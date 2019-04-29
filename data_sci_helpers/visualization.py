import seaborn as sns
from matplotlib import pyplot as plt

def plot_distributions(dataset, features):
    """Plots distributions of a given set of features
    Args:
    dataset: A `pd.DataFrame`, the dataset to shuffle.
    features: A `list` of features
    """
    g = sns.FacetGrid(dataset)
    for feature in features:
        g.map(sns.distplot(dataset[feature]), hist=False, rug=True)

def plot_heatmap(dataset):
    """Plots correlation heatmap
    Args:
    dataset: A `pd.DataFrame`, the dataset to shuffle.
    """
    plt.figure(figsize=(8, 8))
    corr = dataset.corr()
    sns.heatmap(corr, annot=True)

# 3D plots in 2D
def plot_scatter_coolwarm(dataset, feature_1, feature_2, feature_3, title=""):
    """Plots scatterplot of 2 features with the 3rd being represented by a coolwarm color
    Args:
    dataset: A `pd.DataFrame`, the dataset
    feature_1: first feature (x-axis)
    feature_2: second feature (y-axis)
    feature_3: third feature (coolwarm)
    title: Title of the plot
    """    
    plt.figure()
    plt.title(title)
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.scatter(dataset[feature_1], dataset[feature_2], cmap="coolwarm", c=dataset[feature_3] / dataset[feature_3].max())
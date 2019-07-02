from .imports import *
from sklearn.tree import export_graphviz
import graphviz
import os
os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz2.38\\bin"

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

def scatter_combination_of_features(dataset, features):
    """Simple scatterplot of 2 pairs of features
    Args:
    dataset: A `pd.DataFrame`, the dataset
    features: A `list` of features
    """   
    for feature_comb in combinations(features, 2):
        print(feature_comb)
        plt.figure()
        plt.xlabel(feature_comb[0])
        plt.ylabel(feature_comb[1])
        plt.scatter(dataset[feature_comb[0]], dataset[feature_comb[1]])

def plot_2_distributions(data1, data2):
    """Plots 2 seaborn distributions side-by-side
    Args:
    data1: data to create the first distribution from
    data2: data to create the second distribution from
    """
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)
    sns.distplot(data1, ax=ax1)
    sns.distplot(data2, ax=ax2)

def enlarge_font_in_matplotlib(axis, enlarge_to=20):
    """Enlarges plot in a plt's axis
    Source: https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
    Args:
    axis: matplotlib's `Axis` to enlarge font within
    enlarge_to: font size to enlarge to
    """  
    for item in ([axis.title, axis.xaxis.label, axis.yaxis.label] +
             axis.get_xticklabels() + axis.get_yticklabels()):
        item.set_fontsize(enlarge_to)
		
		
# from https://github.com/fastai/fastai/blob/72286b8b22284a53b07d777783ff5d392a3d45b0/old/fastai/structured.py
def draw_tree(t, df, size=10, ratio=0.6, precision=0):
    """ Draws a representation of a random forest in IPython.
    Parameters:
    -----------
    t: The tree you wish to draw
    df: The data used to train the tree. This is used to get the names of the features.
    """
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,
                      special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {',
       f'Tree {{ size={size}; ratio={ratio}', s)))
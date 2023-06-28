import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# graph libraries
import networkx as nx
import sklearn.neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import radius_neighbors_graph
from scipy.spatial import distance
import skimage


def get_2D_distances_knn(positions, id_column, metric='euclidean', k=6):
    """A function to get 2D distances from a set of coordinates using 
    `k` nearest neighbors
    
    Assumes that keys `x` and `y` exist in the data. 

    Parameters
    ----------
    positions (pd.DataFrame):
        A dataframe with positions of nodes and node ids
    id_column (str):
        the column to treat as node ids in the distance matrix
    metric (str):
        A valid distance metric. See `sklearn.metrics.pairwise_distances`.
    k (int):
        The number of nearest neighbors to consider

    Returns
    ----------
    d (pd.DataFrame):
        A 2D distance matrix with entries representing the distance
        between nodes under the metric chosen
    """

    if not 'x' in positions.columns and 'y' in positions.columns:
        raise ValueError("Either `x` or `y` is not in `positions` DataFrame (input).")

    if not id_column in positions.columns:
        raise ValueError(f"id_column: `{id_column}` is not in `positions` DataFrame (input).")

    d = sklearn.neighbors.kneighbors_graph(positions[['x', 'y']], 
                                           n_neighbors=k,
                                           metric=metric,
                                           mode='distance')

    d = d.todense()
    d = pd.DataFrame(d, index=positions[id_column], columns=positions[id_column])
    return d


def get_2D_distances(positions, id_column, metric='euclidean'):
    """A function to get 2D distances from a set of coordinates.
    
    Assumes that keys `x` and `y` exist in the data. 

    Parameters
    ----------
    positions (pd.DataFrame):
        A dataframe with positions of nodes and node ids
    id_column (str):
        the column to treat as node ids in the distance matrix
    metric (str):
        A valid distance metric. See `sklearn.metrics.pairwise_distances`.

    Returns
    ----------
    d (pd.DataFrame):
        A 2D distance matrix with entries representing the distance
        between nodes under the metric chosen
    """

    if not 'x' in positions.columns and 'y' in positions.columns:
        raise ValueError("Either `x` or `y` is not in `positions` DataFrame (input).")

    if not id_column in positions.columns:
        raise ValueError(f"id_column: `{id_column}` is not in `positions` DataFrame (input).")

    d = sklearn.metrics.pairwise_distances(positions[['x', 'y']], metric=metric)
    d = pd.DataFrame(d, index=positions[id_column], columns=positions[id_column])
    return d

def threshold_matrix(A, t, mode='l', binary=True):
    """A function to threhold a matrix based on an input
    value 
    
    Parameters
    ----------
    A (pd.DataFrame or np.array):
        A matrix to threshold
    t (float):
        the threshold value
    mode (str):
        One of: 
            'l' : less than
            'le' : less than or equal to
            'g' : greater than
            'ge' : greater than or equal to
    binary (bool):
        If `True` values more extreme than the threshold
        value are convert to '1'. Otherwise original 
        values are preserved.

    Returns
    ----------
    d (pd.DataFrame):
        A 2D distance matrix with entries representing the distance
        between nodes under the metric chosen
    """

    # @TODO
    pass
        
        

    

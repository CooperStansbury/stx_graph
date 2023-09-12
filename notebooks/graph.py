import pandas as pd 
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import umap
from sklearn.decomposition import PCA

# graph libraries
import networkx as nx
import sklearn.neighbors
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import radius_neighbors_graph
from scipy.spatial import distance
import skimage

def get_flat_distances(points, metric):
    """A function to get pairwise distances, returned for the lower triangle
    indices only 

    Parameters
    ----------
    points (np.array):
        An n dimensional set of points
    metric (str):
        A valid distance metric. See `scipy.spatial.distance.pdist`.
        
    Returns
    ----------
    dist (pd.DataFrame):
        Rows encode indices i, j of the input  and the distance between them
        under the chosen metric
    """
    d = distance.pdist(points, metric=metric)
    i, j = np.triu_indices(points.shape[0], k=1)
    dist = pd.DataFrame({"i" : i,
                         "j" : j,
                         "d" : d})
    return dist
    

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

    d = sklearn.metrics.pairwise_distances(positions[['x', 'y']], 
                                           metric=metric,
                                           force_all_finite=True) # this is the default, as for clarity
    d = pd.DataFrame(d, index=positions[id_column], columns=positions[id_column])
    return d

def closest_node(point, nodes, metric):
    """A function to get the node closest to a point.
    Useful for getting a range of nodes around the center of mass.

    Parameters
    ----------
    point (np.array):
        An n dimensional point
    nodes (np.array):
        A set of nodes with coordinates in n dimensions
    metric (str):
        A valid distance metric. See `scipy.spatial.distance.cdist`.
        
    Returns
    ----------
    closest_index (int):
        The index in the input (`nodes`) of the closest node under the metric chosen.
    """
    closest_index = distance.cdist([point], nodes, metric=metric).argmin()
    return closest_index


def get_neighborhood(df, center=True, n=100, metric='minkowski'):
    """A function to return a neighborhood of n nodes
    around a center point.

    Parameters
    ----------
    df (pd.DataFrame):
        A dataframe with the following columns ['nodeId', 'x', 'y']
    center (bool or array):
        If `True', will find the closest node to the center of mass, 
        else expects 2D array with a point. This point will be
        used as the center of the neigherbood.
    n (int):
        The number of nodes in the neighborhood
    metric (str):
        A distance metric. See scipy.spatial.distance.cdist and 
        sklearn.neighbors.NearestNeighbors

    Returns
    ----------
    Neighborhood (list):
        A list of nodeIds in the specificied neighboorhood
    """
    if isinstance(center, bool):
        # get the node closest to the center
        point = df[['x', 'y']].mean(axis=0)
    else:
        point = center

    # get index of node closest to the point
    # of interest
    nodes = df[['x', 'y']]
    cix = closest_node(point, nodes, metric=metric)

    # treat this point as the center of the neighborhood
    center  = df.loc[cix, 'nodeId']
    
    # get the n points around the center node
    nbrs = NearestNeighbors(n_neighbors=n,  
                            metric=metric, 
                            algorithm='ball_tree').fit(nodes)
    
    distances, indices = nbrs.kneighbors(nodes)

    neighborhood_idx = indices[cix, :]
    neighborhood = df[df.index.isin(neighborhood_idx)]
    return neighborhood['nodeId'].to_list()


# def graph(edges, coordinates):
#     """A function to constuct a graph object given formatted data 
    
#     Parameters
#     ----------
#     edges (pd.DataFrame):
#         Edges dataframe, expected to have the same nodeIds as the coordinates
#     coordinates (pd.DataFrame):
#         The coordinates used for node positions

#     Returns
#     ----------
#     graph (nx.Graph)):
#         A networkx graph object   
#     """
#     pass

    


def build_graph(edges, coordinates, nbrhd=None, threshold=None):
    """A function to build a graph with certain properties
    based on the nodes and edges passed 
    
    Parameters
    ----------
    edges (pd.DataFrame):
        Edges dataframe, expected to have the same nodeIds as the coordinates
    coordinates (pd.DataFrame):
        The coordinates used for node positions
    nbrhd (list):
        A list of nodeIds used to subset the edges and coordinates - may be `None'
    threshold (float):
        A distance threshold on `d` used to define valid edge distances

    Returns
    ----------
    graph (nx.Graph)):
        A networkx graph object
    """
    if not nbrhd is None:
        coordinates = coordinates[coordinates['nodeId'].isin(nbrhd)].reset_index()
        edges = edges[(edges['node1'].isin(nbrhd)) & (edges['node2'].isin(nbrhd))].reset_index()

    # make the positions
    pos = {}
    for idx, row in coordinates.iterrows():
        pos[row['nodeId']] = np.array(row[['x', 'y']])

    # threshold distances
    if not threshold is None:
        edges = edges[edges['d'] < threshold]

    # make the graph object
    G = nx.from_pandas_edgelist(edges, 
                                source='node1',
                                target='node2',
                                edge_attr=True)

    G.pos = pos
    return G


def graph_resistance(G, weight, invert_weights=False, error_mode='raise', tol=1e-9):
    """A function to compute the effective reistance of a graph.
    Note, this is not as flexible as the method `effective_resistance'.
    
    Parameters
    ----------
    G (nx.Graph)
        The graph
    weight (str):
        The attribute used for the edge weight If `None', the graph is unweighted
    invert_weights (bool):
        If `True' the edge weights are taken to be the resistance
        of the edges
    error_mode (str):
        How strict to be for disconnected graphs. Can be one of:
            raise: stop execution when Fiedler number is < than `tol'
            warn: throw a warning, but don't stop execution
            ignore: filter out eigenvalues smaller than the tolerance, not recommended
    tol (float):
        The tolerance for small Fiedler numbers.
        
    Returns
    ----------
    r (float):
        The effective resistance of the graph
    """
    import warnings
    
    H = G.copy()
    node_list = list(H)

    if invert_weights:
        for u, v, d in H.edges(data=True):
            if d[weight] > 0:
                d[weight] = 1 / d[weight]
    
    L = nx.laplacian_matrix(H, node_list, weight=weight)
    L = L.todense()
    N = L.shape[0]
    eval, _ = np.linalg.eigh(L)
    eval = eval[1:]  # drop the zero eigenvalue

    if error_mode == 'raise' and np.min(eval) < tol:
        raise ValueError(f"Small Fiedler number detected! ({np.min(eval)=} < {tol=})")
    elif error_mode == 'warn' and np.min(eval) < tol:
        warnings.warn(f"Small Fiedler number detected! ({np.min(eval)=} < {tol=})")
    elif error_mode == 'ignore' and np.min(eval) < tol:
        eval = eval[eval > tol]
    elif error_mode not in ['raise', 'warn', 'ignore'] and np.min(eval) < tol:
        raise ValueError(f"`error_mode' not recognized: {error_mode}")

    # compute resistance
    r = N * np.sum([1/x for x in eval])
    return r


def laplacian_inverse(G, weight, invert_weights=True):
    """A function to get the inverse of the Laplacian matrix
    for effective resistance calculations.

    Note, this is a convience function
   
    Parameters
    ----------
    G (nx.Graph):
        A graph
    weight (str):
        The edge weight identifier
    invert_weights (bool):
        If `True' the edge weights are taken to be the resistance
        of the edges

    Returns
    ----------
    Q (np.array):
        The Moore-Penrose pseudo-inverse of the Laplacian matrix
    """
    H = G.copy()
    node_list = list(H)
    if invert_weights:
        for u, v, d in H.edges(data=True):
            if d[weight] > 0:
                d[weight] = 1 / d[weight]

    L = nx.laplacian_matrix(H, node_list, weight=weight)
    L = L.todense()
    Q = np.linalg.pinv(L)
    return Q
    
    

def effective_resistance(A, normed=True, tol=1e-9):
    """A function to compute the effective reistance of
    a graph from the adjancency matrix. Will return -1
    for disconnected graphs
    
    Parameters
    ----------
    A (np.array)
        The graph's adjancency matrix
    normed (bool):
        If `True', the Laplacian will be normalized, bounding the
        eigenvalues between 0 and 2
    tol (float):
        The numerical tolerance for zero-valued eigenvalues used
        to determine connectivity of the network

    Returns
    ----------
    r (float):
        The effective resistance of the graph
    """   
    L = scipy.sparse.csgraph.laplacian(A, normed=normed)
    n = L.shape[0]
    eval, _ = np.linalg.eigh(L)
    eval =  eval[1:] # drop the zero-valued eigenvalue!

    # handle the disconnected case
    if eval[0] < tol:
        return -1
    else:
        # compute resistance
        r = n * np.sum([1/x for x in eval])
        return r


def reduce_dim(X, n_components, method='pca', return_reducer=True, **kwargs):
    """A function to reduce the dimension of a scRNA dataset 

    Params:
    --------------
    X (np.array)
        Input dataset. Rows are cells, columns are genes
    n_components (int)
        Number of dimensions in resulting data matrix. Reduces gene-space only
    method (str):
        the method. May be 'pca' or 'umap'
    return_reducer (bool)
        Whether to return the tranformed reducer function
    **kwargs
        Passed to respective reduction functions

    Returns:
    --------------
    E (np.array)
        Output with reduced dimension embedding of genes
    reducer (callable)
        The dimension reducer function 
    """
    methods = ['umap', 'pca']
    if not method in methods:
        raise ValueError(f'`reduce_dim()` method not recognized. Must be in: {methods}')

    if method == 'umap':
        reducer = umap.UMAP(n_components=n_components, **kwargs)
    if method == 'pca':
        reducer = PCA(n_components=n_components, **kwargs)

    
    E = reducer.fit_transform(X) 
    
    if return_reducer:
        return E, reducer
    else:
        return E
        
        
# def threshold_matrix(A, t, mode='l', binary=True):
#     """A function to threhold a matrix based on an input
#     value 
    
#     Parameters
#     ----------
#     A (pd.DataFrame or np.array):
#         A matrix to threshold
#     t (float):
#         the threshold value
#     mode (str):
#         One of: 
#             'l' : less than
#             'le' : less than or equal to
#             'g' : greater than
#             'ge' : greater than or equal to
#     binary (bool):
#         If `True` values more extreme than the threshold
#         value are convert to '1'. Otherwise original 
#         values are preserved.

#     Returns
#     ----------
#     d (pd.DataFrame):
#         A 2D distance matrix with entries representing the distance
#         between nodes under the metric chosen
#     """

#     # @TODO
#     pass
        
        

    

import os
import pandas as pd
import numpy as np
import matplotlib 
import scipy
import matplotlib.pyplot as plt

        
def _normalize_data(X, counts, after=None, copy=False):
    X = X.copy() if copy else X
    if issubclass(X.dtype.type, (int, np.integer)):
        X = X.astype(np.float32)  # TODO: Check if float64 should be used
    else:
        counts_greater_than_zero = counts[counts > 0]

    after = np.median(counts_greater_than_zero, axis=0) if after is None else after
    counts += counts == 0
    counts = counts / after
    if scipy.sparse.issparse(X):
        sparsefuncs.inplace_row_scale(X, 1 / counts)
    elif isinstance(counts, np.ndarray):
        np.divide(X, counts[:, None], out=X)
    else:
        X = np.divide(X, counts[:, None])  # dask does not support kwarg "out"
    return X


def normalize(df, target_sum=1):
    """A function to normalize spots """
    index = df.index
    columns = df.columns
    X = df.to_numpy().copy()
    counts_per_cell = X.sum(1)
    counts_per_cell = np.ravel(counts_per_cell)
    cell_subset = counts_per_cell > 0
    Xnorm = _normalize_data(X, counts_per_cell, target_sum)
    
    ndf = pd.DataFrame(Xnorm, columns=columns, index=index)
    return ndf


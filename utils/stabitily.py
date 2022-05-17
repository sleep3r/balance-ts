import numpy as np


def check_input(Z):
    """ Checks that Z is of the right type and dimension."""
    ### We check that Z is a list or a numpy.array
    if isinstance(Z, list):
        Z = np.asarray(Z)
    elif not isinstance(Z, np.ndarray):
        raise ValueError('The input matrix Z should be of type list or numpy.ndarray')
    ### We check if Z is a matrix (2 dimensions)
    if Z.ndim != 2:
        raise ValueError('The input matrix Z should be of dimension 2')
    return Z


def nogueira(Z: np.ndarray) -> float:
    """
    Let us assume we have M>1 feature sets and d>0 features in total.
    This function computes the stability estimate as given in Definition 4 in  [1].

    Args:
        Z: BINARY matrix (given as a list or as a numpy.ndarray of size M*d).
           Each row of the binary matrix represents a feature set, where a 1 at the f^th position
           means the f^th feature has been selected and a 0 means it has not been selected.

    Returns:
        float: Stability of the feature selection procedure
    """
    Z = check_input(Z)
    M, d = Z.shape
    hatPF = np.mean(Z, axis=0)
    kbar = np.sum(hatPF)
    denom = (kbar / d) * (1 - kbar / d)
    return 1 - (M / (M - 1)) * np.mean(np.multiply(hatPF, 1 - hatPF)) / denom

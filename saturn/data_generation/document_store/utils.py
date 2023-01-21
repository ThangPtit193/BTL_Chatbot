import numpy as np
import bottleneck as bn


def fast_argsort(arrays: np.ndarray, axis=None, top_k: int = 1):
    """Fast argsort for numpy arrays.
    Args:
        arrays: 2D numpy array.
        axis: Axis to sort along.
        top_k: Number of top items to return.

    Returns:
        2D numpy array of indices.
    """
    arrays *= -1
    top_k = min(top_k, arrays.shape[1]) - 1
    indices = np.argpartition(arrays, top_k, axis=axis)
    indices = np.take(indices, np.arange(top_k + 1), axis=axis)
    arrays = np.take_along_axis(arrays, indices, axis=axis)

    # sort within k elements
    ind_part = np.argsort(arrays, axis=axis)
    indices = np.take_along_axis(indices, ind_part, axis=axis)
    return indices


def fast_argsort_bottleneck(arrays: np.ndarray, axis=None, top_k: int = 1):
    # arrays *= -1
    arrays *= -1
    top_k = min(top_k, arrays.shape[1]) - 1
    indices = bn.argpartition(arrays, top_k, axis=axis)
    indices = np.take(indices, np.arange(top_k + 1), axis=axis)
    arrays = np.take_along_axis(arrays, indices, axis=axis)

    # sort within k elements
    ind_part = np.argsort(arrays, axis=axis)
    indices = np.take_along_axis(indices, ind_part, axis=axis)
    return indices


def fast_argsort_1d_bottleneck(arrays: np.ndarray, axis=None, top_k: int = 1):
    # arrays *= -1
    arrays *= -1
    top_k = min(top_k, len(arrays)) - 1
    indices = bn.argpartition(arrays, top_k, axis=axis)
    indices = np.take(indices, np.arange(top_k + 1), axis=axis)
    arrays = np.take_along_axis(arrays, indices, axis=axis)

    # sort within k elements
    ind_part = np.argsort(arrays, axis=axis)
    indices = np.take_along_axis(indices, ind_part, axis=axis)
    return indices

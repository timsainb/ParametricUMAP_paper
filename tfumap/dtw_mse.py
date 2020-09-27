# Written by Kyle McDonald:
#    https://gist.github.com/kylemcdonald/76b6f18fb4026e01196282b59bd31e7e
# based on https://github.com/kylerbrown/ezdtw
# with modifications to be fully njit-able

import numpy as np
from numba import njit


@njit
def sqeuclidean(a, b):
    return np.sum((a - b) ** 2)


@njit
def cdist_jit(a, b):
    na = a.shape[0]
    nb = b.shape[0]
    m = a.shape[1]
    distances = np.empty((na, nb), dtype=a.dtype)
    for i in range(na):
        for j in range(nb):
            distances[i, j] = sqeuclidean(a[i], b[j])
    return np.sqrt(distances)


@njit
def dtw_distance(distances):
    """calculate minimum cumulative distance"""
    DTW = np.empty_like(distances)
    DTW[:, 0] = np.inf
    DTW[0, :] = np.inf
    DTW[0, 0] = 0
    for i in range(1, DTW.shape[0]):
        for j in range(1, DTW.shape[1]):
            DTW[i, j] = distances[i, j] + min(
                DTW[i - 1, j],  # insertion
                DTW[i, j - 1],  # deletion
                DTW[i - 1, j - 1],  # match
            )
    return DTW


@njit
def backtrack(DTW):
    """compute DTW backtrace
    DTW: a matrix of cumulative DTW paths
    returns (p, q): x and y index lists of the optimal DTW path"""
    i, j = DTW.shape[0] - 1, DTW.shape[1] - 1
    p, q = [i], [j]
    while i > 0 and j > 0:
        v0 = DTW[i - 1, j - 1]
        v1 = DTW[i, j - 1]
        v2 = DTW[i - 1, j]
        if v0 <= v1 and v0 <= v2:  # v0 argmin
            i -= 1
            j -= 1
        elif v1 <= v0 and v1 <= v2:  # v1 argmin
            j -= 1
        else:  # v2 argmin
            i -= 1
        p.append(i)
        q.append(j)
    p.reverse()
    q.reverse()
    return p, q


@njit
def dtw(a, b):
    """perform dynamic time warping on two matricies a and b
    first dimension must be time, second dimension shapes must be equal
    
    returns:
        trace_x, trace_y -- the warp path as two lists of indicies. Suitable for use in
        an iterpolation function such as numpy.interp
        
        to warp values from a to b, use: numpy.interp(warpable_values, trace_x, trace_y)
        to warp values from b to a, use: numpy.interp(warpable_values, trace_y, trace_x)
    """
    distance = cdist_jit(a, b)
    cum_min_dist = dtw_distance(distance)
    trace_x, trace_y = backtrack(cum_min_dist)
    return trace_x, trace_y


def build_dtw_mse(shape):
    """
    First build a dtw with `dtw_metric = build_dtw_mse(x[0].shape),
    then umap.UMAP(metric=dtw_metric).fit_transform(x.reshape(len(x), -1))
    """

    @njit
    def dtw_mse(a_flat, b_flat):
        a = a_flat.reshape(*shape)
        b = b_flat.reshape(*shape)
        trace0, trace1 = dtw(a, b)
        # using np.array is easy, but another way might be faster
        aw = a[np.array(trace0)]
        bw = b[np.array(trace1)]
        dist = np.square(aw - bw).mean()
        return dist

    return dtw_mse

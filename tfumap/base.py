"""  This entire section of code could be removed if base UMAP fit() was split into
generate_graph and embed_graph
This is what I did below: create a child of UMAP object and create a fit method
that splits fit() into fit_generate_graph() and embed_graph()
this way, we can create children of this object that only change the embedder. 
"""

from __future__ import print_function

import locale
from warnings import warn
import time

from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.neighbors import KDTree

try:
    import joblib
except ImportError:
    # sklearn.externals.joblib is deprecated in 0.21, will be removed in 0.23
    from sklearn.externals import joblib

import numpy as np
import scipy.sparse
from scipy.sparse import tril as sparse_tril, triu as sparse_triu
import scipy.sparse.csgraph
import numba

import umap.distances as dist

import umap.sparse as sparse
import umap.sparse_nndescent as sparse_nn

from umap import umap_
from umap.umap_ import (
    fuzzy_simplicial_set,
    find_ab_params,
    nearest_neighbors,
    discrete_metric_simplicial_set_intersection,
    general_simplicial_set_intersection,
    reset_local_connectivity,
    make_epochs_per_sample,
    simplicial_set_embedding,
)

from umap.utils import (
    tau_rand_int,
    deheap_sort,
    submatrix,
    ts,
    csr_unique,
    fast_knn_indices,
)
from umap.rp_tree import rptree_leaf_array, make_forest
from umap.nndescent import (
    # make_nn_descent,
    # make_initialisations,
    # make_initialized_nnd_search,
    nn_descent,
    initialized_nnd_search,
    initialise_search,
)
from umap.rp_tree import rptree_leaf_array, make_forest
from umap.spectral import spectral_layout
from umap.utils import deheap_sort, submatrix
from umap.layouts import (
    optimize_layout_euclidean,
    optimize_layout_generic,
    optimize_layout_inverse,
)

try:
    # Use pynndescent, if installed (python 3 only)
    from pynndescent import NNDescent
    from pynndescent.distances import named_distances as pynn_named_distances
    from pynndescent.sparse import sparse_named_distances as pynn_sparse_named_distances

    _HAVE_PYNNDESCENT = True
except ImportError:
    _HAVE_PYNNDESCENT = False

locale.setlocale(locale.LC_NUMERIC, "C")

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf


class UMAP_tensorflow(umap_.UMAP):
    def __init__(self, *args, **kwargs):
        # retrieve everything from base model
        super(UMAP_tensorflow, self).__init__(*args, **kwargs)

    def fit_generate_graph(self, X, y=None):
        """Generate graph to fit X into an embedded space.
        Optionally use y for supervised dimension reduction.
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'.
        y : array, shape (n_samples)
            A target array for supervised dimension reduction. How this is
            handled is determined by parameters UMAP was instantiated with.
            The relevant attributes are ``target_metric`` and
            ``target_metric_kwds``.
        """

        X = check_array(X, dtype=np.float32, accept_sparse="csr", order="C")
        self._raw_data = X

        # Handle all the optional arguments, setting default
        if self.a is None or self.b is None:
            self._a, self._b = find_ab_params(self.spread, self.min_dist)
        else:
            self._a = self.a
            self._b = self.b

        self._initial_alpha = self.learning_rate

        self._validate_parameters()

        # Check if we should unique the data
        # We've already ensured that we aren't in the precomputed case
        if self.unique:
            # check if the matrix is dense
            if self._sparse_data:
                # Call a sparse unique function
                index, inverse, counts = csr_unique(X)
            else:
                index, inverse, counts = np.unique(
                    X,
                    return_index=True,
                    return_inverse=True,
                    return_counts=True,
                    axis=0,
                )[1:4]
            if self.verbose:
                print(
                    "Unique=True -> Number of data points reduced from ",
                    X.shape[0],
                    " to ",
                    X[index].shape[0],
                )
                most_common = np.argmax(counts)
                print(
                    "Most common duplicate is",
                    index[most_common],
                    " with a count of ",
                    counts[most_common],
                )
        # If we aren't asking for unique use the full index.
        # This will save special cases later.
        else:
            index = list(range(X.shape[0]))
            inverse = list(range(X.shape[0]))

        # Error check n_neighbors based on data size
        if X[index].shape[0] <= self.n_neighbors:
            if X[index].shape[0] == 1:
                self.embedding_ = np.zeros(
                    (1, self.n_components)
                )  # needed to sklearn comparability
                return self

            warn(
                "n_neighbors is larger than the dataset size; truncating to "
                "X.shape[0] - 1"
            )
            self._n_neighbors = X[index].shape[0] - 1
        else:
            self._n_neighbors = self.n_neighbors

        # Note: unless it causes issues for setting 'index', could move this to
        # initial sparsity check above
        if self._sparse_data and not X.has_sorted_indices:
            X.sort_indices()

        random_state = check_random_state(self.random_state)

        if self.verbose:
            print("Construct fuzzy simplicial set")

        if self.metric == "precomputed" and self._sparse_data:
            # For sparse precomputed distance matrices, we just argsort the rows to find
            # nearest neighbors. To make this easier, we expect matrices that are
            # symmetrical (so we can find neighbors by looking at rows in isolation,
            # rather than also having to consider that sample's column too).
            print("Computing KNNs for sparse precomputed distances...")
            if sparse_tril(X).getnnz() != sparse_triu(X).getnnz():
                raise ValueError(
                    "Sparse precomputed distance matrices should be symmetrical!"
                )
            if not np.all(X.diagonal() == 0):
                raise ValueError("Non-zero distances from samples to themselves!")
            self._knn_indices = np.zeros((X.shape[0], self.n_neighbors), dtype=np.int)
            self._knn_dists = np.zeros(self._knn_indices.shape, dtype=np.float)
            for row_id in range(X.shape[0]):
                # Find KNNs row-by-row
                row_data = X[row_id].data
                row_indices = X[row_id].indices
                if len(row_data) < self._n_neighbors:
                    raise ValueError(
                        "Some rows contain fewer than n_neighbors distances!"
                    )
                row_nn_data_indices = np.argsort(row_data)[: self._n_neighbors]
                self._knn_indices[row_id] = row_indices[row_nn_data_indices]
                self._knn_dists[row_id] = row_data[row_nn_data_indices]
            self.graph_, self._sigmas, self._rhos = fuzzy_simplicial_set(
                X[index],
                self.n_neighbors,
                random_state,
                "precomputed",
                self._metric_kwds,
                self._knn_indices,
                self._knn_dists,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                True,
                self.verbose,
            )
        # Handle small cases efficiently by computing all distances
        elif X[index].shape[0] < 4096 and not self.force_approximation_algorithm:
            self._small_data = True
            try:
                # sklearn pairwise_distances fails for callable metric on sparse data
                _m = self.metric if self._sparse_data else self._input_distance_func
                dmat = pairwise_distances(X[index], metric=_m, **self._metric_kwds)
            except (ValueError, TypeError) as e:
                # metric is numba.jit'd or not supported by sklearn,
                # fallback to pairwise special

                if self._sparse_data:
                    # Get a fresh metric since we are casting to dense
                    if not callable(self.metric):
                        _m = dist.named_distances[self.metric]
                        dmat = dist.pairwise_special_metric(
                            X[index].toarray(), metric=_m, kwds=self._metric_kwds,
                        )
                    else:
                        dmat = dist.pairwise_special_metric(
                            X[index],
                            metric=self._input_distance_func,
                            kwds=self._metric_kwds,
                        )
                else:
                    dmat = dist.pairwise_special_metric(
                        X[index],
                        metric=self._input_distance_func,
                        kwds=self._metric_kwds,
                    )
            self.graph_, self._sigmas, self._rhos = fuzzy_simplicial_set(
                dmat,
                self._n_neighbors,
                random_state,
                "precomputed",
                self._metric_kwds,
                None,
                None,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                True,
                self.verbose,
            )
        else:
            # Standard case
            self._small_data = False
            # pass string identifier if pynndescent also defines distance metric
            if _HAVE_PYNNDESCENT:
                if self._sparse_data and self.metric in pynn_sparse_named_distances:
                    nn_metric = self.metric
                elif not self._sparse_data and self.metric in pynn_named_distances:
                    nn_metric = self.metric
                else:
                    nn_metric = self._input_distance_func
            else:
                nn_metric = self._input_distance_func
            (self._knn_indices, self._knn_dists, self._rp_forest) = nearest_neighbors(
                X[index],
                self._n_neighbors,
                nn_metric,
                self._metric_kwds,
                self.angular_rp_forest,
                random_state,
                self.low_memory,
                use_pynndescent=True,
                verbose=self.verbose,
            )

            self.graph_, self._sigmas, self._rhos = fuzzy_simplicial_set(
                X[index],
                self.n_neighbors,
                random_state,
                nn_metric,
                self._metric_kwds,
                self._knn_indices,
                self._knn_dists,
                self.angular_rp_forest,
                self.set_op_mix_ratio,
                self.local_connectivity,
                True,
                self.verbose,
            )

            if not _HAVE_PYNNDESCENT:
                self._search_graph = scipy.sparse.lil_matrix(
                    (X[index].shape[0], X[index].shape[0]), dtype=np.int8
                )
                _rows = []
                _data = []
                for i in self._knn_indices:
                    _non_neg = i[i >= 0]
                    _rows.append(_non_neg)
                    _data.append(np.ones(_non_neg.shape[0], dtype=np.int8))

                self._search_graph.rows = _rows
                self._search_graph.data = _data
                self._search_graph = self._search_graph.maximum(
                    self._search_graph.transpose()
                ).tocsr()

                if (self.metric != "precomputed") and (len(self._metric_kwds) > 0):
                    # Create a partial function for distances with arguments
                    _distance_func = self._input_distance_func
                    _dist_args = tuple(self._metric_kwds.values())
                    if self._sparse_data:

                        @numba.njit()
                        def _partial_dist_func(ind1, data1, ind2, data2):
                            return _distance_func(ind1, data1, ind2, data2, *_dist_args)

                        self._input_distance_func = _partial_dist_func
                    else:

                        @numba.njit()
                        def _partial_dist_func(x, y):
                            return _distance_func(x, y, *_dist_args)

                        self._input_distance_func = _partial_dist_func

        # Currently not checking if any duplicate points have differing labels
        # Might be worth throwing a warning...

        if y is not None:
            len_X = len(X) if not self._sparse_data else X.shape[0]
            if len_X != len(y):
                raise ValueError(
                    "Length of x = {len_x}, length of y = {len_y}, while it must be equal.".format(
                        len_x=len_X, len_y=len(y)
                    )
                )
            y_ = check_array(y, ensure_2d=False)[index]
            if self.target_metric == "categorical":
                if self.target_weight < 1.0:
                    far_dist = 2.5 * (1.0 / (1.0 - self.target_weight))
                else:
                    far_dist = 1.0e12
                self.graph_ = discrete_metric_simplicial_set_intersection(
                    self.graph_, y_, far_dist=far_dist
                )
            elif self.target_metric in dist.DISCRETE_METRICS:
                if self.target_weight < 1.0:
                    scale = 2.5 * (1.0 / (1.0 - self.target_weight))
                else:
                    scale = 1.0e12
                # self.graph_ = discrete_metric_simplicial_set_intersection(
                #     self.graph_,
                #     y_,
                #     metric=self.target_metric,
                #     metric_kws=self.target_metric_kwds,
                #     metric_scale=scale
                # )

                metric_kws = dist.get_discrete_params(y_, self.target_metric)

                self.graph_ = discrete_metric_simplicial_set_intersection(
                    self.graph_,
                    y_,
                    metric=self.target_metric,
                    metric_kws=metric_kws,
                    metric_scale=scale,
                )
            else:
                if len(y_.shape) == 1:
                    y_ = y_.reshape(-1, 1)
                if self.target_n_neighbors == -1:
                    target_n_neighbors = self._n_neighbors
                else:
                    target_n_neighbors = self.target_n_neighbors

                # Handle the small case as precomputed as before
                if y.shape[0] < 4096:
                    try:
                        ydmat = pairwise_distances(
                            y_, metric=self.target_metric, **self._target_metric_kwds
                        )
                    except (TypeError, ValueError):
                        ydmat = dist.pairwise_special_metric(
                            y_,
                            metric=self.target_metric,
                            kwds=self._target_metric_kwds,
                        )

                    target_graph, target_sigmas, target_rhos = fuzzy_simplicial_set(
                        ydmat,
                        target_n_neighbors,
                        random_state,
                        "precomputed",
                        self._target_metric_kwds,
                        None,
                        None,
                        False,
                        1.0,
                        1.0,
                        False,
                    )
                else:
                    # Standard case
                    target_graph, target_sigmas, target_rhos = fuzzy_simplicial_set(
                        y_,
                        target_n_neighbors,
                        random_state,
                        self.target_metric,
                        self._target_metric_kwds,
                        None,
                        None,
                        False,
                        1.0,
                        1.0,
                        False,
                    )
                # product = self.graph_.multiply(target_graph)
                # # self.graph_ = 0.99 * product + 0.01 * (self.graph_ +
                # #                                        target_graph -
                # #                                        product)
                # self.graph_ = product
                print("COMPUTING SIMPLICIAL SET INTERSECTION")
                self.graph_ = general_simplicial_set_intersection(
                    self.graph_, target_graph, self.target_weight
                )
                self.graph_ = reset_local_connectivity(self.graph_)
        return index, inverse

    def fit_embed_data(self, X, y, index, inverse):

        if isinstance(self.init, np.ndarray):
            init = check_array(self.init, dtype=np.float32, accept_sparse=False)
        else:
            init = self.init

        if self.n_epochs is None:
            n_epochs = 0
        else:
            n_epochs = self.n_epochs

        if self.verbose:
            print(ts(), "Construct embedding")

        random_state = check_random_state(self.random_state)

        self.embedding_ = simplicial_set_embedding(
            self._raw_data[index],  # JH why raw data?
            self.graph_,
            self.n_components,
            self._initial_alpha,
            self._a,
            self._b,
            self.repulsion_strength,
            self.negative_sample_rate,
            n_epochs,
            init,
            random_state,
            self._input_distance_func,
            self._metric_kwds,
            self._output_distance_func,
            self._output_metric_kwds,
            self.output_metric in ("euclidean", "l2"),
            self.random_state is None,
            self.verbose,
        )[inverse]

        if self.verbose:
            print(ts() + " Finished embedding")

        self._input_hash = joblib.hash(self._raw_data)

    def fit(self, X, y=None):
        """
        """
        # create graph and parameters
        index, inverse = self.fit_generate_graph(X, y)
        # perform fit
        self.fit_embed_data(X, y, index, inverse)
        return self

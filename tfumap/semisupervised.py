import numpy as np
import tensorflow as tf
from tfumap.umap import compute_cross_entropy

from pynndescent import NNDescent
from scipy.sparse import csr_matrix
from sklearn.utils import check_random_state, check_array
from umap.umap_ import fuzzy_simplicial_set, discrete_metric_simplicial_set_intersection
from scipy import optimize
from functools import partial

random_state = check_random_state(None)


def build_fuzzy_simplicial_set(X, y=None, n_neighbors=15):
    """
    Build nearest neighbor graph, then fuzzy simplicial set

    Parameters
    ----------
    X : [type]
        [description]
    n_neighbors : int, optional
        [description], by default 15
    """
    n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
    n_iters = max(5, int(round(np.log2(X.shape[0]))))

    # get nearest neighbors
    nnd = NNDescent(
        X,
        n_neighbors=n_neighbors,
        metric="euclidean",
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
    )

    # get indices and distances
    knn_indices, knn_dists = nnd.neighbor_graph

    random_state = check_random_state(None)
    # build graph
    umap_graph, sigmas, rhos = fuzzy_simplicial_set(
        X=X,
        n_neighbors=n_neighbors,
        metric="euclidean",
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )

    if y is not None:
        # set far_dist based on the assumption that target_weight == 1
        far_dist = 1.0e12
        y_ = check_array(y, ensure_2d=False)
        umap_graph = discrete_metric_simplicial_set_intersection(
            umap_graph, y_, far_dist=far_dist
        )

    return umap_graph


def convert_distance_to_probability(distances, a, b):
    """ convert distance representation into probability, 
        as a function of a, b params
    """
    return 1.0 / (1.0 + a * distances ** (2 * b))


def find_a_b(min_dist=0.1):
    """ determine optimal params a, b to such that distances less than 
        min_dist have a probability of zero
    """
    # input distances
    x = np.linspace(0, 3, 300)
    # optimal output (if close enough, don't try to make closer)
    y = np.exp(-x + min_dist) * (x > min_dist) + (x < min_dist)

    # run through scipy,optimize a, b parameters for min_dist
    # get the optimal
    (a, b), _ = optimize.curve_fit(f=convert_distance_to_probability, xdata=x, ydata=y)

    # a and b parameters for computing probability in low-d
    a = tf.constant(a, dtype=tf.float32,)
    b = tf.constant(b, dtype=tf.float32,)
    return a, b


def compute_classifier_loss(X, y, encoder, classifier, sparse_ce):
    """ compute the cross entropy loss for classification
        """
    d = classifier(encoder(X))
    return sparse_ce(y, d)


def compute_umap_loss(
    batch_to,
    batch_from,
    embedder,
    encoder,
    _a,
    _b,
    negative_sample_rate=5,
    repulsion_strength=1,
):
    """
        compute the cross entropy loss for learning embeddings

        Parameters
        ----------
        batch_to : tf.int or tf.float32
            Either X or the index locations of the embeddings for verticies (to)
        batch_from : tf.int or tf.float32
            Either X or the index locations of the embeddings for verticies (from)

        Returns
        -------
        ce_loss : tf.float
            cross entropy loss for UMAP
        embedding_to : tf.float
            embeddings for verticies (to)
        embedding_from : tf.float
            embeddings for verticies (from)
        """

    # encode
    embedding_to = embedder(encoder(batch_to))
    embedding_from = embedder(encoder(batch_from))

    # get negative samples
    embedding_neg_to = tf.repeat(embedding_to, negative_sample_rate, axis=0)
    repeat_neg = tf.repeat(embedding_from, negative_sample_rate, axis=0)
    embedding_neg_from = tf.gather(
        repeat_neg, tf.random.shuffle(tf.range(tf.shape(repeat_neg)[0]))
    )

    #  distances between samples
    distance_embedding = tf.concat(
        [
            tf.norm(embedding_to - embedding_from, axis=1),
            tf.norm(embedding_neg_to - embedding_neg_from, axis=1),
        ],
        axis=0,
    )

    # convert probabilities to distances
    probabilities_distance = convert_distance_to_probability(distance_embedding, _a, _b)

    # treat positive samples as p=1, and negative samples as p=0
    probabilities_graph = tf.concat(
        [tf.ones(embedding_to.shape[0]), tf.zeros(embedding_neg_to.shape[0])], axis=0,
    )

    # cross entropy loss
    (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
        probabilities_graph,
        probabilities_distance,
        repulsion_strength=repulsion_strength,
    )

    return (
        attraction_loss,
        repellant_loss,
        ce_loss,
    )


def batch_epoch_edges(edges_to, edges_from, batch_size):
    """ permutes and batches edges for epoch
        """
    # compute the number of batches in one epoch
    n_batches = int(len(edges_to) / batch_size)
    # permute list of edges
    permutation_mask = np.random.permutation(len(edges_to))[: n_batches * batch_size]
    to_all = tf.reshape(tf.gather(edges_to, permutation_mask), (n_batches, batch_size))
    from_all = tf.reshape(
        tf.gather(edges_from, permutation_mask), (n_batches, batch_size)
    )
    # return a tensorflow dataset of one epoch's worth of batches
    return tf.data.Dataset.from_tensor_slices((to_all, from_all))


def create_edge_iterator(
    head, tail, weight, batch_size, max_sample_repeats_per_epoch=25
):
    """ create an iterator for edges
    """
    # set the maximum number of times each edge should be repeated per epoch
    epochs_per_sample = np.clip(
        (weight / np.max(weight)) * max_sample_repeats_per_epoch,
        1,
        max_sample_repeats_per_epoch,
    ).astype("int")

    edges_to_exp, edges_from_exp = (
        np.array([np.repeat(head, epochs_per_sample.astype("int"))]),
        np.array([np.repeat(tail, epochs_per_sample.astype("int"))]),
    )
    edge_iter = tf.data.Dataset.from_tensor_slices((edges_to_exp, edges_from_exp))
    edge_iter = edge_iter.repeat()
    # edge_iter = edge_iter.map(batch_epoch_edges)
    edge_iter = edge_iter.map(partial(batch_epoch_edges, batch_size=batch_size))
    edge_iter = edge_iter.prefetch(buffer_size=10)

    return iter(edge_iter), np.shape(edges_to_exp)[1]


def create_classification_iterator(X_labeled, y_labeled, batch_size):
    """
    Creates a tensorflow iterator for classification data (X, y)
    """
    #
    # create labeled data iterator
    labeled_data = tf.data.Dataset.from_tensor_slices((X_labeled, y_labeled))
    labeled_data = labeled_data.repeat()
    labeled_data = labeled_data.shuffle(np.min([len(y_labeled), 1000]))
    labeled_data = labeled_data.batch(batch_size)
    labeled_data = labeled_data.prefetch(buffer_size=1)

    return iter(labeled_data)


def create_validation_iterator(valid_X, valid_Y, batch_size):
    """ Create an iterator that returns validation X and Y
    """
    data_valid = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y))
    data_valid = data_valid.cache()
    data_valid = data_valid.batch(batch_size)
    data_valid = data_valid.prefetch(buffer_size=1)

    return data_valid, len(valid_X)

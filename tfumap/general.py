import numpy as np
import tensorflow as tf
from umap.umap_ import make_epochs_per_sample
from scipy.sparse import csr_matrix


def convert_distance_to_probability(distances, a, b):
    """ convert distance representation into probability, 
        as a function of a, b params
    """
    return 1.0 / (1.0 + a * distances ** (2 * b))


def compute_cross_entropy(
    probabilities_graph, probabilities_distance, EPS=1e-4, negative_sample_scale=1.0
):
    """ Compute cross entropy between low and high probability
    """
    # cross entropy
    attraction_term = -probabilities_graph * tf.math.log(
        tf.clip_by_value(probabilities_distance, EPS, 1.0)
    )
    repellant_term = (
        -(1.0 - probabilities_graph)
        * tf.math.log(tf.clip_by_value(1.0 - probabilities_distance, EPS, 1.0))
        * negative_sample_scale
    )

    # balance the expected losses between atrraction and repel
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE


def remove_redundant_edges(graph):

    edges = graph.nonzero()
    # remove redundancies
    redundant_edges = edges[0] < edges[1]
    edges = (edges[0][redundant_edges], edges[1][redundant_edges])
    # reconstruct graph
    graph = csr_matrix(
        (np.array(graph[edges]).flatten(), (edges[0], edges[1])), shape=graph.shape
    )

    return graph


def get_graph_elements(graph_, n_epochs):

    graph_ = remove_redundant_edges(graph_)

    ### should we remove redundancies here??
    graph = graph_.tocoo()
    graph.sum_duplicates()

    n_vertices = graph.shape[1]

    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200

    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices

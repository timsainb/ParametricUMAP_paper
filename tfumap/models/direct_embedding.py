# base umap embedding with tensorflow
# Author: Tim Sainburg

import tensorflow as tf
import numpy as np
from tfumap.base import UMAP_tensorflow
from tqdm.autonotebook import tqdm

from tfumap.general import (
    convert_distance_to_probability,
    compute_cross_entropy,
    get_graph_elements,
)

from datetime import datetime
from sklearn.utils import check_random_state
from sklearn.neighbors import KDTree
import joblib
from umap.utils import ts
from umap.spectral import spectral_layout


class UMAP_tensorflow_direct_embedding(UMAP_tensorflow):
    def __init__(
        self,
        optimizer=tf.keras.optimizers.Adadelta(50),
        tensorboard_logdir=None,
        batch_size=None,
        max_epochs_per_sample=10,
        negative_sample_rate=2,
        training_epochs=200,
        *args,
        **kwargs
    ):
        # retrieve everything from base model
        super(UMAP_tensorflow_direct_embedding, self).__init__(*args, **kwargs)

        self.batch_size = (
            batch_size  # number of elements per batch for tensorflow embedding
        )
        self.max_epochs_per_sample = (
            max_epochs_per_sample  # maximum number of repeated edges during training
        )
        #   (higher = more memory load, lower = less difference between low and high probabilility)
        self.optimizer = optimizer

        self.random_state = check_random_state(None)

        # number of training epochs. Different than n_epochs for base model
        #   (which is still used, e.g. for embedding using original method)
        self.training_epochs = training_epochs

        # log summary data for tensorboard
        if tensorboard_logdir == None:
            self.tensorboard_logdir = "/tmp/tensorboard/" + datetime.now().strftime(
                "%Y%m%d-%H%M%S"
            )
        else:
            self.tensorboard_logdir = tensorboard_logdir
        self.summary_writer = tf.summary.create_file_writer(self.tensorboard_logdir)

    def init_embedding_from_graph(self, init="spectral"):
        if isinstance(init, str) and init == "random":
            embedding = self.random_state.uniform(
                low=-10.0, high=10.0, size=(graph.shape[0], self.n_components)
            ).astype(np.float32)
        elif isinstance(init, str) and init == "spectral":
            # We add a little noise to avoid local minima for optimization to come
            print("self.metric_kwds", self.metric_kwds, self.metric)
            initialisation = spectral_layout(
                self._raw_data,
                self.graph_,
                self.n_components,
                self.random_state,
                metric=self.metric,
                metric_kwds=self.metric_kwds,
            )
            expansion = 10.0 / np.abs(initialisation).max()
            embedding = (initialisation * expansion).astype(
                np.float32
            ) + self.random_state.normal(
                scale=0.0001, size=[self.graph_.shape[0], self.n_components]
            ).astype(
                np.float32
            )

        else:
            init_data = np.array(init)
            if len(init_data.shape) == 2:
                if np.unique(init_data, axis=0).shape[0] < init_data.shape[0]:
                    tree = KDTree(init_data)
                    dist, ind = tree.query(init_data, k=2)
                    nndist = np.mean(dist[:, 1])
                    embedding = init_data + self.random_state.normal(
                        scale=0.001 * nndist, size=init_data.shape
                    ).astype(np.float32)
                else:
                    embedding = init_data

        return embedding

    def batch_epoch_edges(self, edges_to, edges_from):
        """ permutes and batches edges for epoch
        """

        n_batches = int(len(edges_to) / self.batch_size)
        permutation_mask = np.random.permutation(len(edges_to))[
            : n_batches * self.batch_size
        ]
        to_all = tf.reshape(
            tf.gather(edges_to, permutation_mask), (n_batches, self.batch_size)
        )
        from_all = tf.reshape(
            tf.gather(edges_from, permutation_mask), (n_batches, self.batch_size)
        )
        return tf.data.Dataset.from_tensor_slices((to_all, from_all))

    def create_edge_iterator(self, head, tail, epochs_per_sample):
        """ create an iterator for edges
        """

        epochs_per_sample = np.clip(
            (epochs_per_sample / np.max(epochs_per_sample))
            * self.max_epochs_per_sample,
            1,
            self.max_epochs_per_sample,
        ).astype("int")

        edges_to_exp, edges_from_exp = (
            np.array([np.repeat(head, epochs_per_sample)]),
            np.array([np.repeat(tail, epochs_per_sample)]),
        )
        edge_iter = tf.data.Dataset.from_tensor_slices((edges_to_exp, edges_from_exp))
        edge_iter = edge_iter.repeat()
        edge_iter = edge_iter.map(self.batch_epoch_edges)
        # edge_iter = edge_iter.cache()
        edge_iter = edge_iter.prefetch(buffer_size=10)

        return iter(edge_iter)

    def compute_umap_loss(self, batch_to, batch_from):
        """
        compute the cross entropy loss for learning embeddings
        """
        # get the embeddings
        embedding_to = tf.gather(self.embedding, batch_to)
        embedding_from = tf.gather(self.embedding, batch_from)

        # embeddings for negative samples
        embedding_neg_to = tf.gather(
            self.embedding, tf.repeat(batch_to, self.negative_sample_rate)
        )
        embedding_neg_from = tf.gather(
            self.embedding,
            tf.random.shuffle(tf.repeat(batch_from, self.negative_sample_rate)),
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
        probabilities_distance = convert_distance_to_probability(
            distance_embedding, self._a, self._b
        )

        # treat positive samples as p=1, and negative samples as p=0
        probabilities_graph = tf.concat(
            [tf.ones(embedding_to.shape[0]), tf.zeros(embedding_neg_to.shape[0])],
            axis=0,
        )

        # cross entropy loss
        (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
            probabilities_graph, probabilities_distance, negative_sample_scale=1.0,
        )

        return ce_loss

    @tf.function
    def train(self, batch_to, batch_from):
        with tf.GradientTape() as tape:
            ce_loss = self.compute_umap_loss(batch_to, batch_from)

        grads = tape.gradient(ce_loss, [self.embedding])
        grads = [tf.clip_by_value(grad, -4.0, 4.0) * self.alpha for grad in grads]
        self.optimizer.apply_gradients(zip(grads, [self.embedding]))

        return ce_loss

    def embed_data(self, X, y, index, inverse, **kwargs):

        # get data from graph
        graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(
            self.graph_, self.n_epochs
        )

        # get embedding
        embedding = self.init_embedding_from_graph(**kwargs)
        embedding = embedding[index]

        if self.batch_size is None:
            self.batch_size = int(n_vertices / 1)

        print("batch_size", self.batch_size)

        # alpha is a hack for circumventing tensorflow's problem with sparse vectors
        self.alpha = tf.Variable(1.0)

        #### convert embedding to Variable
        self.embedding = tf.Variable(embedding.astype(np.float32, order="C"))

        # create iterator for edges
        edge_iter = self.create_edge_iterator(head, tail, epochs_per_sample)

        if self.verbose:
            print(ts(), "Embedding with TensorFlow")

        iter_ = zip(edge_iter, np.arange(self.training_epochs))
        if self.verbose:
            iter_ = tqdm(iter_, desc="epoch", total=self.training_epochs)

        for edge_epoch, epoch in iter_:
            # loop through batches
            for batch_to, batch_from in edge_epoch:
                ce_loss = self.train(batch_to, batch_from)
                with self.summary_writer.as_default():
                    tf.summary.scalar("loss/train", tf.reduce_mean(ce_loss), step=epoch)
                    self.summary_writer.flush()
        self.summary_writer.close()

        if self.verbose:
            print(ts() + " Finished embedding")

        self.embedding_ = self.embedding.numpy()[inverse]

        self._input_hash = joblib.hash(self._raw_data)

# base umap embedding with tensorflow
# Author: Tim Sainburg

import tensorflow as tf
import numpy as np
from tfumap.base import UMAP_tensorflow
from tqdm.autonotebook import tqdm

tf.get_logger().setLevel("INFO")

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


class UMAP_neural_network(UMAP_tensorflow):
    def __init__(
        self,
        optimizer=None,
        tensorboard_logdir=None,
        batch_size=None,  # size of batch used for batch training
        dims=None,  # dimensionality of data, if not flat (e.g. images for ConvNet)
        negative_sample_rate=2,  # how many negative samples per positive samples for training
        max_epochs_per_sample=None,  # (setting this value to 1 is equivalent to computing UMAP on nearest neighbors graph without fuzzy_simplicial_set)
        direct_embedding=False,  # whether to learn embeddings directly, or use neural network
        encoder=None,  # the neural net used for encoding (defaults to 3 layer 100 neuron fc)
        decoder=None,  # the neural net used for decoding (defaults to 3 layer 100 neuron fc)
        training_epochs=50,  # number of epochs to train for
        decoding_method=None,  # how to decode "autoencoder", "network", or None
        valid_X=None,  # validation data for reconstruction and classification error
        valid_Y=None,  # validation labels for reconstruction and classification error
        *args,
        **kwargs
    ):
        # retrieve everything from base model
        super(UMAP_neural_network, self).__init__(*args, **kwargs)

        self.batch_size = batch_size

        self.max_epochs_per_sample = (
            max_epochs_per_sample  # maximum number of repeated edges during training
        )

        # set optimizer, Adam is better for neural networks, Adadelta is better for direct embedding

        if optimizer is None:
            if direct_embedding:
                self.optimizer = tf.keras.optimizers.Adadelta(50)
            else:
                self.optimizer = tf.keras.optimizers.Adam(1e-3)
        else:
            self.optimizer = optimizer

        self.random_state = check_random_state(None)

        self.dims = dims  # if this is an image, we should reshape for network
        self.encoder = encoder  # neural network used for embedding
        self.decoder = decoder  # neural network used for decoding
        # whether to use a decoder, and what type  ("autoencoder", "network", None)
        self.decoding_method = decoding_method

        # whether to encode with encoder, or embed directly
        self.direct_embedding = direct_embedding

        # if (self.decoding_method is not None) & direct_embedding:
        #    # TODO this could be supported if we passed X to the iterator alongside
        #    raise NotImplementedError(
        #        "Reconstruction network not supported for direct embedding"
        #    )

        self.valid_X = (
            valid_X  # validation data used for reconstruction or classification
        )
        self.valid_Y = valid_Y  # validation labels used for classification

        # make a binary cross entropy object for reconstruction
        if self.decoding_method in ["autoencoder", "network"]:
            self.binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()

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
        self.summary_writer_train = tf.summary.create_file_writer(
            self.tensorboard_logdir + "/train"
        )
        self.summary_writer_valid = tf.summary.create_file_writer(
            self.tensorboard_logdir + "/valid"
        )

    def compute_umap_loss(self, batch_to, batch_from):
        """
        compute the cross entropy loss for learning embeddings
        """

        if self.direct_embedding:
            # get the embeddings
            embedding_to = tf.gather(self.embedding, batch_to)
            embedding_from = tf.gather(self.embedding, batch_from)
        else:
            # encode
            embedding_to = self.encoder(batch_to)
            embedding_from = self.encoder(batch_from)

        # get negative samples
        embedding_neg_to = tf.repeat(embedding_to, self.negative_sample_rate, axis=0)
        repeat_neg = tf.repeat(embedding_from, self.negative_sample_rate, axis=0)
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
            probabilities_graph, probabilities_distance, negative_sample_scale=1.0
        )

        return ce_loss, embedding_to, embedding_from

    def compute_reconstruction_loss(self, X, Z=None):
        """
        [summary]

        Parameters
        ----------
        X : [type]
            input data
        Z : [type], optional
            embedding of input data, by default None

        Returns
        -------
        reonstruction_loss
            binary cross entropy loss for reconstruction
        """
        if Z is None:
            if self.direct_embedding:
                if self.embedding_ is None:
                    raise ValueError(
                        "Cannot reconstruct when direct_embedding = True without first embedding"
                    )
            else:
                Z = self.encoder(X)
        X_r = self.decoder(Z)
        reconstruction_loss = self.binary_cross_entropy(X, X_r)
        return reconstruction_loss

    @tf.function
    def train(self, batch_to, batch_from):
        with tf.GradientTape() as tape:
            # all networks get UMAP loss
            ce_loss, embedding_to, embedding_from = self.compute_umap_loss(
                batch_to, batch_from
            )

            if self.decoding_method == "autoencoder":

                # grab X data if using a direct embedding
                if self.direct_embedding:
                    # get the embeddings
                    batch_to = tf.gather(self._raw_data, batch_to)
                    batch_from = tf.gather(self._raw_data, batch_from)

                ce_loss = tf.reduce_mean(ce_loss)
                reconstruction_loss = tf.reduce_mean(
                    [
                        self.compute_reconstruction_loss(batch_to, embedding_to),
                        self.compute_reconstruction_loss(batch_from, embedding_from),
                    ]
                )
            elif self.decoding_method == "network":
                # grab X data if using a direct embedding
                if self.direct_embedding:
                    # get the embeddings
                    batch_to = tf.gather(self._raw_data, batch_to)
                    batch_from = tf.gather(self._raw_data, batch_from)

                ce_loss = tf.reduce_mean(ce_loss)
                # the same loss as autoencoder, but stopping the gradient before the embedding
                reconstruction_loss = tf.reduce_mean(
                    [
                        self.compute_reconstruction_loss(
                            batch_to, tf.stop_gradient(embedding_to)
                        ),
                        self.compute_reconstruction_loss(
                            batch_from, tf.stop_gradient(embedding_from)
                        ),
                    ]
                )
            else:
                reconstruction_loss = 0.0

            loss = ce_loss + reconstruction_loss

            grads = tape.gradient(loss, self.trainable_variables)
            grads = [tf.clip_by_value(grad, -4.0, 4.0) * self.alpha for grad in grads]
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return ce_loss, reconstruction_loss

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

        if self.max_epochs_per_sample is not None:
            epochs_per_sample = np.clip(
                (epochs_per_sample / np.max(epochs_per_sample))
                * self.max_epochs_per_sample,
                1,
                self.max_epochs_per_sample,
            ).astype("int")

        edges_to_exp, edges_from_exp = (
            np.array([np.repeat(head, epochs_per_sample.astype("int"))]),
            np.array([np.repeat(tail, epochs_per_sample.astype("int"))]),
        )
        edge_iter = tf.data.Dataset.from_tensor_slices((edges_to_exp, edges_from_exp))
        edge_iter = edge_iter.repeat()
        edge_iter = edge_iter.map(self.batch_epoch_edges)
        edge_iter = edge_iter.prefetch(buffer_size=10)

        return iter(edge_iter), np.shape(edges_to_exp)[1]

    def init_embedding_from_graph(self, init="spectral"):
        """ Initialize embedding using graph

        Parameters
        ----------
        init : str, optional
            [description], by default "spectral"

        Returns
        -------
        [type]
            [description]
        """
        if isinstance(init, str) and init == "random":
            embedding = self.random_state.uniform(
                low=-10.0, high=10.0, size=(graph.shape[0], self.n_components)
            ).astype(np.float32)
        elif isinstance(init, str) and init == "spectral":
            # We add a little noise to avoid local minima for optimization to come

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

    def prepare_networks(self):

        if self.direct_embedding:
            self.trainable_variables = [self.embedding]
        else:
            # create an encoder network, if one does not exist
            if self.encoder is None:
                self.encoder = tf.keras.Sequential()
                self.encoder.add(tf.keras.layers.InputLayer(input_shape=self.dims))
                self.encoder.add(tf.keras.layers.Flatten())
                self.encoder.add(tf.keras.layers.Dense(units=100, activation="relu"))
                self.encoder.add(tf.keras.layers.Dense(units=100, activation="relu"))
                self.encoder.add(tf.keras.layers.Dense(units=100, activation="relu"))
                self.encoder.add(tf.keras.layers.Dense(units=self.n_components))

            # get list of trainable variables for gradient descent
            self.trainable_variables = self.encoder.trainable_variables

        if (self.decoding_method in ["autoencoder", "network"]) & (
            self.decoder is None
        ):
            self.decoder = tf.keras.Sequential()
            self.decoder.add(tf.keras.layers.InputLayer(input_shape=self.n_components))
            self.decoder.add(tf.keras.layers.Flatten())
            self.decoder.add(tf.keras.layers.Dense(units=100, activation="relu"))
            self.decoder.add(tf.keras.layers.Dense(units=100, activation="relu"))
            self.decoder.add(tf.keras.layers.Dense(units=100, activation="relu"))
            self.decoder.add(
                tf.keras.layers.Dense(units=np.product(self.dims), activation="sigmoid")
            )
            self.decoder.add(tf.keras.layers.Reshape(self.dims))

        if self.decoding_method in ["autoencoder", "network"]:
            self.trainable_variables += self.decoder.trainable_variables

    def create_validation_iterator(self):
        """ Create an iterator that returns validation X and Y
        """
        # create a Y validation dataset if one doesn't exist
        if self.valid_Y is None:
            self.valid_Y = np.zeros(len(self.valid_X)) - 1

        data_valid = tf.data.Dataset.from_tensor_slices((self.valid_X, self.valid_Y))
        data_valid = data_valid.cache()
        data_valid = data_valid.batch(self.batch_size)
        data_valid = data_valid.prefetch(buffer_size=1)

        return data_valid, len(self.valid_X)

    def embed_data(self, X, y, index, inverse, **kwargs):

        # get data from graph
        graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(
            self.graph_, self.training_epochs
        )

        # number of elements per batch for tensorflow embedding
        if self.batch_size is None:
            # batch size can be larger if its just over embeddings
            if self.direct_embedding & (self.decoding_method is None):
                self.batch_size = np.min([n_vertices, 60000])
            else:
                self.batch_size = np.min([n_vertices, 1000])

        print("batch_size", self.batch_size)

        # get embedding initialization if embedding directly
        if self.direct_embedding:
            embedding = self.init_embedding_from_graph(**kwargs)
            embedding = embedding[index]
            self.embedding = tf.Variable(embedding.astype(np.float32, order="C"))

        # alpha is a hack for circumventing tensorflow's bug with sparse vectors
        #   this is only needed for the adadelta on direct embeddings
        self.alpha = tf.Variable(1.0)

        # get dimensions of data
        if self.dims is None:
            self.dims = [np.shape(X)[-1]]

        # create networks, if one does not exist
        self.prepare_networks()

        # set a batch size, if one does not exist
        if self.batch_size is None:
            self.batch_size = 100

        # create iterator for data/edges
        edge_iter, n_edges_per_epoch = self.create_edge_iterator(
            head, tail, epochs_per_sample
        )

        # get batches per epoch
        n_batches_per_epoch = int(np.ceil(n_edges_per_epoch / self.batch_size))

        # create an iterator for validation data
        if (
            self.decoding_method in ["autoencoder", "network"]
            and self.valid_X is not None
        ):
            data_valid, n_valid_samp = self.create_validation_iterator()
            # number of batches corresponding to one epoch
            n_valid_batches_per_epoch = int(n_valid_samp / self.batch_size)
        if self.verbose:
            print(ts(), "Embedding with TensorFlow")

        # create keras summary objects for loss
        train_loss_umap_summary = tf.keras.metrics.Mean(
            "train_loss_umap", dtype=tf.float32
        )
        if self.decoding_method in ["autoencoder", "network"]:
            train_loss_recon_summary = tf.keras.metrics.Mean(
                "train_loss_recon", dtype=tf.float32
            )
            if self.valid_X is not None:
                valid_loss_recon_summary = tf.keras.metrics.Mean(
                    "valid_loss_recon", dtype=tf.float32
                )

        # create a tqdm iterator to show epoch progress
        if self.verbose:
            epoch_iter = tqdm(desc="epoch", total=self.training_epochs)

        batch = 0
        for edge_epoch, epoch in zip(edge_iter, np.arange(self.training_epochs)):

            if self.verbose & (n_batches_per_epoch > 100):
                edge_tqdm = tqdm(desc="batch", total=n_batches_per_epoch, leave=False)
            # loop through batches

            # edge_epoch = edge_epoch.map(gather_X).prefetch(buffer_size=5)

            for batch_to, batch_from in edge_epoch:
                batch += 1
                # if this is a direct encoding, the embeddings should be used directly
                if self.direct_embedding:
                    ce_loss, reconstruction_loss = self.train(batch_to, batch_from)
                    # ce_loss, reconstruction_loss = self.train(batch_to, batch_from)
                else:
                    ce_loss, reconstruction_loss = self.train(
                        X[batch_to], X[batch_from]
                    )
                train_loss_umap_summary(ce_loss)
                if self.decoding_method in ["autoencoder", "network"]:
                    train_loss_recon_summary(reconstruction_loss)

                if self.verbose & (n_batches_per_epoch > 100):
                    edge_tqdm.update(1)

                # save summary information
                with self.summary_writer_train.as_default():
                    tf.summary.scalar(
                        "umap_loss", train_loss_umap_summary.result(), step=batch
                    )
                    if self.decoding_method in ["autoencoder", "network"]:
                        tf.summary.scalar(
                            "recon_loss", train_loss_recon_summary.result(), step=batch,
                        )
                    self.summary_writer_train.flush()

            if self.verbose:
                if n_batches_per_epoch > 100:
                    # close tqdm iterator
                    edge_tqdm.update(edge_tqdm.total - edge_tqdm.n)
                    edge_tqdm.close()
                epoch_iter.update(1)

            # compute test loss for reconstruction
            if (
                self.decoding_method in ["autoencoder", "network"]
                and self.valid_X is not None
                and self.direct_embedding is False
            ):
                for valid_batch_X, valid_batch_Y in iter(data_valid):
                    # get loss for reconstruction
                    valid_recon_loss = tf.reduce_mean(
                        self.compute_reconstruction_loss(valid_batch_X)
                    )
                    valid_loss_recon_summary(valid_recon_loss)

                # save summary information
                with self.summary_writer_valid.as_default():
                    tf.summary.scalar(
                        "recon_loss", valid_loss_recon_summary.result(), step=batch,
                    )
                    self.summary_writer_valid.flush()

        # self.summary_writer.close()

        if self.verbose:
            print(ts() + " Finished embedding")

        # make embedding as projected batch

        if self.direct_embedding:
            self.embedding_ = self.embedding.numpy()[inverse]
        else:
            self.embedding = self.transform(X[index])
            self.embedding_ = self.embedding[inverse]

        self._input_hash = joblib.hash(self._raw_data)

    def original_transform(self, X):
        """Run the original transform method from UMAP, rather than neural
        network based.
        """
        return super().transform(X)

    def original_inverse_transform(self, X):
        """Run the original inverse_transform method from UMAP, rather than neural
        network based.
        """
        return super().inverse_transform(X)

    def transform(self, X):
        """Transform X into the existing embedded space and return that
        transformed output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            New data to be transformed.
        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the new data in low-dimensional space.
        """
        # embed using nearest neighbors method if embedding without a network
        if self.direct_embedding:
            return self.original_transform(X)

        n_batches = np.ceil(len(X) / self.batch_size).astype(int)

        if len(self.dims) > 1:
            X = X.reshape(X, [len(X)] + self.dims)

        projections = []
        for batch in np.arange(n_batches):
            projections.append(
                self.encoder(
                    X[(batch * self.batch_size) : ((batch + 1) * self.batch_size)]
                ).numpy()
            )
        projections = np.vstack(projections)
        return projections

    def inverse_transform(self, X):
        """Transform X in the existing embedded space back into the input
        data space and return that transformed output.
        Parameters
        ----------
        X : array, shape (n_samples, n_components)
            New points to be inverse transformed.
        Returns
        -------
        X_new : array, shape (n_samples, n_features)
            Generated data points new data in data space.
        """

        if self.decoding_method in ["autoencoder", "network"]:
            n_batches = np.ceil(len(X) / self.batch_size).astype(int)

            projections = []
            for batch in np.arange(n_batches):
                projections.append(
                    self.decoder(
                        X[(batch * self.batch_size) : ((batch + 1) * self.batch_size)]
                    ).numpy()
                )
            projections = np.vstack(projections)
            return projections

        else:
            return self.original_inverse_transform(X)

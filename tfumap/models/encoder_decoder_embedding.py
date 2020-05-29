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


class UMAP_neural_network(UMAP_tensorflow):
    def __init__(
        self,
        optimizer=tf.keras.optimizers.Adam(1e-3),
        tensorboard_logdir=None,
        batch_size=None,
        dims=None,
        max_epochs_per_sample=10,
        negative_sample_rate=2,
        encoder=None,
        decoder=None,
        training_epochs=50,
        decoding_method=None,  # "autoencoder", "network", or None
        valid_X=None,
        valid_Y=None,
        *args,
        **kwargs
    ):
        # retrieve everything from base model
        super(UMAP_neural_network, self).__init__(*args, **kwargs)

        self.batch_size = (
            batch_size  # number of elements per batch for tensorflow embedding
        )
        self.max_epochs_per_sample = (
            max_epochs_per_sample  # maximum number of repeated edges during training
        )
        #   (higher = more memory load, lower = less difference between low and high probabilility)
        self.optimizer = optimizer

        self.random_state = check_random_state(None)

        self.dims = dims  # if this is an image, we should reshape for network
        self.encoder = encoder  # neural network used for embedding
        self.decoder = decoder  # neural network used for decoding
        # whether to use a decoder, and what type  ("autoencoder", "network", None)
        self.decoding_method = decoding_method
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
        # encode
        embedding_to = self.encoder(batch_to)
        embedding_from = self.encoder(batch_from)

        # get negative samples
        embedding_neg_to = tf.repeat(embedding_to, self._n_neighbors, axis=0)
        repeat_neg = tf.repeat(embedding_from, self._n_neighbors, axis=0)
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
            Z = self.encoder(X)
        X_r = self.decoder(Z)
        reconstruction_loss = self.binary_cross_entropy(X, X_r)
        return reconstruction_loss

    # @tf.function
    def train(self, batch_to, batch_from):
        with tf.GradientTape() as tape:
            if self.decoding_method == "autoencoder":
                ce_loss, embedding_to, embedding_from = self.compute_umap_loss(
                    batch_to, batch_from
                )
                reconstruction_loss = tf.reduce_mean(
                    [
                        self.compute_reconstruction_loss(batch_to, embedding_to),
                        self.compute_reconstruction_loss(batch_from, embedding_from),
                    ]
                )

                loss = tf.reduce_mean(ce_loss) + tf.reduce_mean(reconstruction_loss)
            else:

                ce_loss, embedding_to, embedding_from = self.compute_umap_loss(
                    batch_to, batch_from
                )
                reconstruction_loss = None
                loss = ce_loss
        if self.decoding_method == "autoencoder":
            grads = tape.gradient(
                loss,
                self.encoder.trainable_variables + self.decoder.trainable_variables,
            )
            grads = [tf.clip_by_value(grad, -4.0, 4.0) for grad in grads]
            self.optimizer.apply_gradients(
                zip(
                    grads,
                    self.encoder.trainable_variables + self.decoder.trainable_variables,
                )
            )
        else:
            grads = tape.gradient(loss, self.encoder.trainable_variables)
            grads = [tf.clip_by_value(grad, -4.0, 4.0) for grad in grads]
            self.optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))

        # "network" implies the decoder is not trained jointly with the encoder
        if self.decoding_method == "network":
            with tf.GradientTape() as tape:
                reconstruction_loss = tf.reduce_mean(
                    [
                        self.compute_reconstruction_loss(batch_to, embedding_to),
                        self.compute_reconstruction_loss(batch_from, embedding_from),
                    ]
                )
            grads = tape.gradient(reconstruction_loss, self.decoder.trainable_variables)
            grads = [tf.clip_by_value(grad, -4.0, 4.0) for grad in grads]
            self.optimizer.apply_gradients(zip(grads, self.decoder.trainable_variables))

        return ce_loss, reconstruction_loss

    def create_edge_iterator(self, X, head, tail, epochs_per_sample):
        """ create an iterator for data/edges
            """

        epochs_per_sample = np.clip(
            (epochs_per_sample / np.max(epochs_per_sample))
            * self.max_epochs_per_sample,
            1,
            self.max_epochs_per_sample,
        ).astype("int")

        # repeat data (this is a little memory intensive)
        edges_to = np.repeat(head, epochs_per_sample)
        edges_from = np.repeat(tail, epochs_per_sample)

        X_to = X[edges_to].reshape([len(edges_to)] + self.dims)
        X_from = X[edges_from].reshape([len(edges_from)] + self.dims)

        # create tensorflow dataset, generating an epoch of batches
        data_train = tf.data.Dataset.from_tensor_slices((X_to, X_from))
        data_train = data_train.repeat()
        data_train = data_train.shuffle(self.batch_size * 5)
        # data_train = data_train.cache()
        data_train = data_train.batch(self.batch_size)
        data_train = data_train.prefetch(buffer_size=5)

        return iter(data_train), len(X_to)

    def prepare_networks(self):

        # create an encoder network, if one does not exist
        if self.encoder is None:
            self.encoder = tf.keras.Sequential()
            self.encoder.add(tf.keras.layers.InputLayer(input_shape=self.dims))
            self.encoder.add(tf.keras.layers.Flatten())
            self.encoder.add(tf.keras.layers.Dense(units=100, activation="relu"))
            self.encoder.add(tf.keras.layers.Dense(units=100, activation="relu"))
            self.encoder.add(tf.keras.layers.Dense(units=100, activation="relu"))
            self.encoder.add(tf.keras.layers.Dense(units=self.n_components))

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

        # get dimensions of data
        if self.dims is None:
            self.dims = [np.shape(X)[-1]]

        # create networks, if one does not exist
        self.prepare_networks()

        # set a batch size, if one does not exist
        if self.batch_size is None:
            self.batch_size = 100

        # create iterator for data/edges
        edge_iter, n_samp_per_epoch = self.create_edge_iterator(
            X, head, tail, epochs_per_sample
        )

        # number of batches corresponding to one epoch
        n_batches_per_epoch = int(n_samp_per_epoch / self.batch_size)

        #
        if (
            self.decoding_method in ["autoencoder", "network"]
            and self.valid_X is not None
        ):
            data_valid, n_valid_samp = self.create_validation_iterator()
            # number of batches corresponding to one epoch
            n_valid_batches_per_epoch = int(n_valid_samp / self.batch_size)
        if self.verbose:
            print(ts(), "Embedding with TensorFlow")

        # create iterator for batch loop
        iter_ = zip(edge_iter, np.arange(n_batches_per_epoch * self.training_epochs))
        if self.verbose:
            epoch_iter = tqdm(total=self.training_epochs, desc="epoch")
            iter_ = tqdm(
                iter_, desc="batch", total=self.training_epochs * n_batches_per_epoch
            )

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

        # loop through batches
        epoch = 0
        for (batch_to, batch_from), batch in iter_:
            # loop through batches
            ce_loss, reconstruction_loss = self.train(batch_to, batch_from)
            train_loss_umap_summary(ce_loss)
            if self.decoding_method in ["autoencoder", "network"]:
                train_loss_recon_summary(reconstruction_loss)

            with self.summary_writer_train.as_default():
                tf.summary.scalar(
                    "umap_loss", train_loss_umap_summary.result(), step=batch
                )
                if self.decoding_method in ["autoencoder", "network"]:
                    tf.summary.scalar(
                        "recon_loss", train_loss_recon_summary.result(), step=batch,
                    )
                self.summary_writer_train.flush()

            if (batch % n_batches_per_epoch == 0) & (batch != 0):
                epoch_iter.update(1)
                epoch += 1

                # compute test loss for reconstruction
                if (
                    self.decoding_method in ["autoencoder", "network"]
                    and self.valid_X is not None
                ):
                    for valid_batch_X, valid_batch_Y in iter(data_valid):
                        # TODO get loss for reconstruction
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

        # complete iterator
        epoch_iter.update(self.training_epochs - epoch_iter.n)
        # self.summary_writer.close()

        if self.verbose:
            print(ts() + " Finished embedding")

        # make embedding as projected batch
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

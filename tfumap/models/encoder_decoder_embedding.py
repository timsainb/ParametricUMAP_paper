# base umap embedding with tensorflow
# Author: Tim Sainburg

import tensorflow as tf
import numpy as np
from tfumap.base import UMAP_tensorflow
from tqdm.autonotebook import tqdm
import tempfile

tf.get_logger().setLevel("INFO")

from umap.umap_ import make_epochs_per_sample
from scipy.sparse import csr_matrix

from datetime import datetime
from sklearn.utils import check_random_state
from sklearn.neighbors import KDTree
import joblib
from umap.utils import ts
from umap.spectral import spectral_layout


class tfUMAP(UMAP_tensorflow):
    def __init__(
        self,
        optimizer=None,
        tensorboard_logdir=None,
        batch_size=None,
        dims=None,
        negative_sample_rate=2,
        max_epochs_per_sample=None,
        direct_embedding=False,
        train_classifier=False,
        encoder=None,
        decoder=None,
        classifier=None,
        training_epochs=50,
        decoding_method=None,
        valid_X=None,
        valid_Y=None,
        classification_loss_weight=1.0,
        umap_loss_weight=1.0,
        reconstruction_loss_weight=1.0,
        *args,
        **kwargs
    ):
        """
        [summary]

        Parameters
        ----------
        optimizer : tf.keras.optimizers, optional
            The tensorflow optimizer used for embedding. Usually ADAM for neural nets
            and adadelta for direct optimization, by default None
        tensorboard_logdir : str, optional
            Location to place tensorboard logs, by default None
        batch_size : int, optional
            size of batch used for batch training, by default None
        dims : tuple, optional
            dimensionality of data, if not flat (e.g. images for ConvNet), by default None
        negative_sample_rate : int, optional
            How many negative samples per positive samples for training, by default 2
        max_epochs_per_sample : [type], optional
            maximum number of times each edge can be sampled for one epoch (setting this value to
             1 is equivalent to computing UMAP on nearest neighbors graph without 
             fuzzy_simplicial_set), by default None
        direct_embedding : bool, optional
            whether to learn embeddings directly, or use neural network, by default False
        train_classifier : bool, optional
            whether a classifier network should be jointly trained with data, by default False
        encoder : tf.keras.Sequential, optional
            the neural net used for encoding (defaults to 3 layer 100 neuron fc), by default None
        decoder : tf.keras.Sequential, optional
            the neural net used for decoding (defaults to 3 layer 100 neuron fc), by default None
        classifier : tf.keras.Sequential, optional
            the neural net used for decoding (defaults to 3 layer 100 neuron fc), by default None
        training_epochs : int, optional
            number of epochs to train for. Seperate from n_epochs, which here is used for determining
            the number of training samples per edge based on probability., by default 50
        decoding_method : string, optional (default None)
            how to decode. Options are:
                * "autoencoder"
                * "network"
                * None
        valid_X : [type], optional
            validation data for reconstruction and classification error, by default None
        valid_Y : [type], optional
            validation labels for reconstruction and classification error, by default None
        classification_loss_weight : float, optional
            weight for the classification loss, by default 1.0
        umap_loss_weight : float, optional
            weight for the umap loss, by default 1.0
        reconstruction_loss_weight : float, optional
            weight for the reconstruction loss, by default 1.0
        """
        # retrieve everything from base model
        super(tfUMAP, self).__init__(*args, **kwargs)

        self.batch_size = batch_size

        self.max_epochs_per_sample = (
            max_epochs_per_sample  # maximum number of repeated edges during training
        )
        self.train_classifier = train_classifier
        self.classifier = classifier
        if self.train_classifier:
            self.compute_sparse_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )
        # set weights for each loss component
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.umap_loss_weight = umap_loss_weight
        self.classification_loss_weight = classification_loss_weight

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
            self.tensorboard_logdir = os.path.join(
                tempfile.gettempdir(),
                "tensorboard",
                datetime.now().strftime("%Y%m%d-%H%M%S"),
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

    def compute_classifier_loss(self, X, y):
        """ compute the cross entropy loss for classification
        """
        # get the encoder output (before embedding)
        base_input = self.encoder_base(X)
        # get predictions for class
        predictions = self.classifier(base_input)
        loss = self.compute_sparse_cross_entropy(tf.expand_dims(y, -1), predictions)
        acc = tf.keras.metrics.sparse_categorical_accuracy(
            tf.expand_dims(y, -1), predictions
        )
        return loss, tf.reduce_mean(acc)

    @tf.function
    def train_batch(self, batch_to, batch_from, X=None, y=None):
        with tf.GradientTape() as tape:
            # all methods get UMAP loss
            umap_loss, embedding_to, embedding_from = self.compute_umap_loss(
                batch_to, batch_from
            )
            # get reconstruction loss if applicable
            if self.decoding_method == "autoencoder":

                # grab X data if using a direct embedding
                if self.direct_embedding:
                    # get the embeddings
                    batch_to = tf.gather(self._raw_data, batch_to)
                    batch_from = tf.gather(self._raw_data, batch_from)

                umap_loss = tf.reduce_mean(umap_loss)
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

                umap_loss = tf.reduce_mean(umap_loss)
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
            # get classifier loss if applicable
            if self.train_classifier:
                classifier_loss, classifier_acc = self.compute_classifier_loss(X, y)
            else:
                classifier_loss, classifier_acc = 0.0, 0.0
            loss = (
                umap_loss * self.umap_loss_weight
                + reconstruction_loss * self.reconstruction_loss_weight
                + classifier_loss * self.classification_loss_weight
            )

            grads = tape.gradient(loss, self.trainable_variables)
            grads = [tf.clip_by_value(grad, -4.0, 4.0) * self.alpha for grad in grads]
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return umap_loss, reconstruction_loss, classifier_loss, classifier_acc

    def batch_epoch_edges(self, edges_to, edges_from):
        """ permutes and batches edges for epoch
        """
        # compute the number of batches in one epoch
        n_batches = int(len(edges_to) / self.batch_size)
        # permute list of edges
        permutation_mask = np.random.permutation(len(edges_to))[
            : n_batches * self.batch_size
        ]
        to_all = tf.reshape(
            tf.gather(edges_to, permutation_mask), (n_batches, self.batch_size)
        )
        from_all = tf.reshape(
            tf.gather(edges_from, permutation_mask), (n_batches, self.batch_size)
        )
        # return a tensorflow dataset of one epoch's worth of batches
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

    def init_embedding_from_graph(self, graph, init="spectral"):
        """ Initialize embedding using graph. This is for direct embeddings. 

        Parameters
        ----------
        init : str, optional
            Type of initialization to use. Either random, or spectral, by default "spectral"

        Returns
        -------
        embedding : np.array
            the initialized embedding
        """
        if isinstance(init, str) and init == "random":
            embedding = self.random_state.uniform(
                low=-10.0, high=10.0, size=(graph.shape[0], self.n_components)
            ).astype(np.float32)
        elif isinstance(init, str) and init == "spectral":
            # We add a little noise to avoid local minima for optimization to come

            initialisation = spectral_layout(
                self._raw_data,
                graph,
                self.n_components,
                self.random_state,
                metric=self.metric,
                metric_kwds=self._metric_kwds,
            )
            expansion = 10.0 / np.abs(initialisation).max()
            embedding = (initialisation * expansion).astype(
                np.float32
            ) + self.random_state.normal(
                scale=0.0001, size=[graph.shape[0], self.n_components]
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
                self.encoder.add(
                    tf.keras.layers.Dense(units=self.n_components, name="z")
                )

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

        if self.train_classifier:
            # get name of final layer before embedding for encoder
            last_layer = self.encoder.layers[-2].name
            last_layer_shape = self.encoder.get_layer(last_layer).output.shape[1:]
            # subset all layers of the encoder but the embedding
            self.encoder_base = tf.keras.models.Model(
                [self.encoder.inputs[0]], [self.encoder.get_layer(last_layer).output]
            )

            if self.classifier is None:
                self.classifier = tf.keras.Sequential()
                self.classifier.add(
                    tf.keras.layers.InputLayer(input_shape=last_layer_shape)
                )
                self.classifier.add(tf.keras.layers.Flatten())
                self.classifier.add(tf.keras.layers.Dense(units=100, activation="relu"))
                self.classifier.add(tf.keras.layers.Dense(units=100, activation="relu"))
                self.classifier.add(tf.keras.layers.Dense(units=100, activation="relu"))
                self.classifier.add(
                    tf.keras.layers.Dense(
                        units=self.n_classes, activation="softmax", name="predictions"
                    )
                )
                self.trainable_variables += self.classifier.trainable_variables

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

    def create_classification_iterator(self, X_labeled, y_labeled):
        # Creates a tensorflow iterator for classification data (X, y)
        # create labeled data iterator
        self.labeled_data = tf.data.Dataset.from_tensor_slices((X_labeled, y_labeled))
        self.labeled_data = self.labeled_data.repeat()
        self.labeled_data = self.labeled_data.shuffle(np.min([len(y_labeled), 1000]))
        self.labeled_data = self.labeled_data.batch(self.batch_size)
        self.labeled_data = self.labeled_data.prefetch(buffer_size=1)

        return iter(self.labeled_data)

    def embed_data(self, X, y, index, inverse, **kwargs):

        # get data from graph
        graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(
            self.graph_, self.n_epochs
        )

        # number of elements per batch for tensorflow embedding
        if self.batch_size is None:
            # batch size can be larger if its just over embeddings
            if self.direct_embedding & (self.decoding_method is None):
                self.batch_size = np.min([n_vertices, 60000])
            else:
                self.batch_size = np.min([n_vertices, 1000])

        # get embedding initialization if embedding directly
        if self.direct_embedding:
            embedding = self.init_embedding_from_graph(graph, **kwargs)
            embedding = embedding[index]
            self.embedding = tf.Variable(embedding.astype(np.float32, order="C"))

        # alpha is a hack for circumventing tensorflow's bug with sparse vectors
        #   this is only needed for the adadelta on direct embeddings
        self.alpha = tf.Variable(1.0)

        # get dimensions of data
        if self.dims is None:
            self.dims = [np.shape(X)[-1]]
        # reshape data for network
        if self.dims is not None:
            if len(self.dims) > 1:
                X = np.reshape(X, [len(X)] + list(self.dims))
                if self.valid_X is not None:
                    self.valid_X = np.reshape(
                        self.valid_X, [len(self.valid_X)] + list(self.dims)
                    )

        # if network is jointly training a classifier, get labeled data
        if (y is not None) & self.train_classifier:
            # get the number of training classes
            label_mask = y != -1
            # subset labeled X and Y
            X_labeled = X[label_mask]
            y_labeled = y[label_mask]
            self.n_classes = len(np.unique(y_labeled))

        # create networks, if one does not exist
        self.prepare_networks()

        # set a batch size, if one does not exist
        if self.batch_size is None:
            self.batch_size = 100

        # create iterator for data/edges
        edge_iter, n_edges_per_epoch = self.create_edge_iterator(
            head, tail, epochs_per_sample
        )

        # if network is jointly training a classifier, prepare data iterator
        if (y is not None) & self.train_classifier:
            # generate tensorflow iterator for classifier labels
            labeled_iter = self.create_classification_iterator(
                self, X_labeled, y_labeled
            )

        # get batches per epoch
        n_batches_per_epoch = int(np.ceil(n_edges_per_epoch / self.batch_size))

        # create an iterator for validation data
        if (
            self.decoding_method in ["autoencoder", "network"]
            or (self.train_classifier)
        ) and self.valid_X is not None:
            data_valid, n_valid_samp = self.create_validation_iterator()
            # number of batches corresponding to one epoch
            n_valid_batches_per_epoch = int(n_valid_samp / self.batch_size)
        if self.verbose:
            print(ts(), "Embedding with TensorFlow")

        # create keras summary objects for loss
        self.create_summary_metrics()

        # create a tqdm iterator to show epoch progress
        if self.verbose:
            epoch_iter = tqdm(desc="epoch", total=self.training_epochs)

        batch = 0
        X_lab, y_lab = None, None  # default classifier values
        for edge_epoch, epoch in zip(edge_iter, np.arange(self.training_epochs)):

            if self.verbose & (n_batches_per_epoch > 200):
                edge_tqdm = tqdm(desc="batch", total=n_batches_per_epoch, leave=False)

            # loop through batches
            for batch_to, batch_from in edge_epoch:
                batch += 1
                # if training a classifier, get X and y data
                if self.train_classifier:
                    X_lab, y_lab = labeled_iter.next()

                # if this is a direct encoding, the embeddings should be used directly
                if self.direct_embedding:
                    (
                        ce_loss,
                        reconstruction_loss,
                        classifier_loss,
                        classifier_acc,
                    ) = self.train_batch(batch_to, batch_from, X_lab, y_lab)
                else:
                    (
                        ce_loss,
                        reconstruction_loss,
                        classifier_loss,
                        classifier_acc,
                    ) = self.train_batch(X[batch_to], X[batch_from], X_lab, y_lab)
                # save losses to tensorflow summary
                self.summary_metrics["train_loss_umap"](ce_loss)
                if self.decoding_method in ["autoencoder", "network"]:
                    self.summary_metrics["train_loss_recon"](reconstruction_loss)
                if self.train_classifier:
                    self.summary_metrics["train_loss_classif"](classifier_loss)
                    self.summary_metrics["train_acc_classif"](classifier_acc)
                if self.verbose & (n_batches_per_epoch > 200):
                    edge_tqdm.update(1)

                # save summary information
                with self.summary_writer_train.as_default():
                    tf.summary.scalar(
                        "umap_loss",
                        self.summary_metrics["train_loss_umap"].result(),
                        step=batch,
                    )
                    if self.decoding_method in ["autoencoder", "network"]:
                        tf.summary.scalar(
                            "recon_loss",
                            self.summary_metrics["train_loss_recon"].result(),
                            step=batch,
                        )

                    if self.train_classifier:
                        tf.summary.scalar(
                            "classif_loss",
                            self.summary_metrics["train_loss_classif"].result(),
                            step=batch,
                        )
                        tf.summary.scalar(
                            "classif_acc",
                            self.summary_metrics["train_acc_classif"].result(),
                            step=batch,
                        )

                    self.summary_writer_train.flush()

            # update tqdm iterators
            if self.verbose:
                if n_batches_per_epoch > 200:
                    # close tqdm iterator
                    edge_tqdm.update(edge_tqdm.total - edge_tqdm.n)
                    edge_tqdm.close()
                epoch_iter.update(1)

            # compute test loss for reconstruction and classification
            if self.valid_X is not None and self.direct_embedding is False:
                for valid_batch_X, valid_batch_Y in iter(data_valid):
                    # get loss for reconstruction
                    if self.decoding_method in ["autoencoder", "network"]:

                        valid_recon_loss = tf.reduce_mean(
                            self.compute_reconstruction_loss(valid_batch_X)
                        )
                        self.summary_metrics["valid_loss_recon"](valid_recon_loss)

                    # get loss for accuracy
                    if self.train_classifier:
                        classifier_loss, classifier_acc = self.compute_classifier_loss(
                            valid_batch_X, valid_batch_Y
                        )
                        self.summary_metrics["valid_loss_classif"](classifier_loss)
                        self.summary_metrics["valid_acc_classif"](classifier_acc)
                # save summary information

                with self.summary_writer_valid.as_default():
                    if self.decoding_method in ["autoencoder", "network"]:
                        tf.summary.scalar(
                            "recon_loss",
                            self.summary_metrics["valid_loss_recon"].result(),
                            step=batch,
                        )
                    if self.train_classifier:
                        tf.summary.scalar(
                            "classif_loss",
                            self.summary_metrics["valid_loss_classif"].result(),
                            step=batch,
                        )
                        tf.summary.scalar(
                            "classif_acc",
                            self.summary_metrics["valid_acc_classif"].result(),
                            step=batch,
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

    def create_summary_metrics(self):
        """
        Creates a set of tf.keras summary metrics to store loss information
        """
        # create keras summary objects for loss
        self.summary_metrics = {}
        self.summary_metrics["train_loss_umap"] = tf.keras.metrics.Mean(
            "train_loss_umap", dtype=tf.float32
        )
        if self.decoding_method in ["autoencoder", "network"]:
            self.summary_metrics["train_loss_recon"] = tf.keras.metrics.Mean(
                "train_loss_recon", dtype=tf.float32
            )
            if self.valid_X is not None:
                self.summary_metrics["valid_loss_recon"] = tf.keras.metrics.Mean(
                    "valid_loss_recon", dtype=tf.float32
                )

        if self.train_classifier:
            self.summary_metrics["train_loss_classif"] = tf.keras.metrics.Mean(
                "train_loss_classif", dtype=tf.float32
            )
            self.summary_metrics["valid_loss_classif"] = tf.keras.metrics.Mean(
                "valid_loss_classif", dtype=tf.float32
            )
            self.summary_metrics["train_acc_classif"] = tf.keras.metrics.Mean(
                "train_acc_classif", dtype=tf.float32
            )
            self.summary_metrics["valid_acc_classif"] = tf.keras.metrics.Mean(
                "valid_acc_classif", dtype=tf.float32
            )

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
            X = np.reshape(X, [len(X)] + list(self.dims))

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


def convert_distance_to_probability(distances, a, b):
    """
    convert (euclidean) distance in low-d into probability, 
        as a function of a, b params. 

    Parameters
    ----------
    distances : float
        euclidean sitances
    a : int
        the a/alpha UMAP parameter
    b : int
        the b/beta UMAP parameter

    Returns
    -------
    probabilities float
        the edge probability in low-d space
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
    redundant_edges = edges[0] <= edges[1]
    edges = (edges[0][redundant_edges], edges[1][redundant_edges])
    # reconstruct graph
    graph = csr_matrix(
        (np.array(graph[edges]).flatten(), (edges[0], edges[1])), shape=graph.shape
    )
    return graph


def get_graph_elements(graph_, n_epochs):
    ### should we remove redundancies () here??
    # graph_ = remove_redundant_edges(graph_)

    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()
    # number of vertices in dataset
    n_vertices = graph.shape[1]
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    # get epochs per sample based upon edge probability
    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices

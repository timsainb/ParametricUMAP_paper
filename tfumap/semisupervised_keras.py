from tfumap.load_datasets import load_CIFAR10, mask_labels
import tensorflow as tf
from tfumap.paths import MODEL_DIR
import numpy as np


pretrained_networks = {
    "cifar10_old": {
        "augmented": {
            4: "cifar10_4____2020_08_09_22_16_45_780732_baseline_augmented",  # 15
            16: "cifar10_16____2020_08_09_22_43_34_001017_baseline_augmented",  # 29
            64: "cifar10_64____2020_08_09_22_16_13_299376_baseline_augmented",  # 47
            256: "cifar10_256____2020_08_09_22_05_58_228942_baseline_augmented",  # 71
            1024: "cifar10_1024____2020_08_09_23_20_57_619002_baseline_augmented",  # 84
            "full": "cifar10_full____2020_08_09_22_03_51_910408_baseline_augmented",  # ~91.5
        },
        "not_augmented": {
            4: "cifar10_4____2020_08_09_22_13_28_904367_baseline",  # 0.1542
            16: "cifar10_16____2020_08_09_21_58_11_099843_baseline",  # 0.2335
            64: "cifar10_64____2020_08_09_22_14_33_994011_baseline",  # 0.3430
            256: "cifar10_128____2020_08_09_21_58_00_869329_baseline",  # 0.4693
            1024: "cifar10_1024____2020_08_09_22_14_06_923244_baseline",  # 0.7864
            "full": "cifar10_full____2020_08_09_21_54_03_152503_baseline",  # 0.8923
        },
    },
    "cifar10": {
        "augmented": {
            4: "cifar10_4____2020_08_11_19_25_58_939055_baseline_augmented",  # 15 # 26
            16: "cifar10_16____2020_08_11_19_25_49_190428_baseline_augmented",  # 29 # 40
            64: "cifar10_64____2020_08_11_19_25_41_266466_baseline_augmented",  # 47 # ~60
            256: "cifar10_256____2020_08_11_19_25_33_546350_baseline_augmented",  # 71 # 76
            1024: "cifar10_1024____2020_08_11_19_25_33_541963_baseline_augmented",  # 84 # 86
            "full": "cifar10_full____2020_08_11_19_25_33_543821_baseline_augmented",  # ~91.5 # ~93
        },
        "not_augmented": {
            4: "cifar10_4____2020_08_17_16_02_48_330135_baseline",  # 0.1542 # 21
            16: "cifar10_16____2020_08_17_15_08_23_820108_baseline",  # 0.2335 # 30
            64: "cifar10_64____2020_08_17_15_08_38_407886_baseline",  # 0.3430 # 50
            256: "cifar10_256____2020_08_17_22_16_16_108071_baseline",  # 0.4693 # 72
            1024: "cifar10_1024____2020_08_17_15_08_08_912644_baseline",  # 0.7864 # 84
            "full": "cifar10_full____2020_08_17_15_08_15_778694_baseline",  # 0.8923 # 90
        },
        "umap_augmented": {
            4: "cifar10_0.0_4____2020_08_19_00_40_15_425455_umap_augmented",  #
            16: "cifar10_0.0_16____2020_08_19_00_40_13_037112_umap_augmented",
            64: "cifar10_0.0_64____2020_08_19_00_40_13_032397_umap_augmented",
            256: "cifar10_0.0_256____2020_08_18_16_16_47_694512_umap_augmented",
            1024: "cifar10_0.0_1024____2020_08_19_10_25_26_973224_umap_augmented",
            "full": "cifar10_0.0_full____2020_08_19_00_40_18_936212_umap_augmented",
        },
        # "umap_not_augmented": {
        #    4: "cifar10_0.0_4____2020_08_19_10_31_18_319944_umap_augmented",
        #    16: "cifar10_0.0_16____2020_08_19_10_32_02_165996_umap_augmented",
        #    64: "cifar10_0.0_64____2020_08_19_10_32_58_283124_umap_augmented",
        #    256: "cifar10_0.0_256____2020_08_19_10_34_27_343116_umap_augmented",
        #    1024: "cifar10_0.0_1024____2020_08_19_10_35_26_231624_umap_augmented",
        #    "full": "",
        # },
        "umap_not_augmented": {
            4: "cifar10_0.0_4____2020_08_19_14_35_43_127626_umap_augmented",  #
            16: "cifar10_0.0_16____2020_08_19_14_35_43_867336_umap_augmented",
            64: "cifar10_0.0_64____2020_08_19_14_35_43_736036_umap_augmented",
            256: "cifar10_0.0_256____2020_08_19_14_38_31_105228_umap_augmented",
            1024: "cifar10_0.0_1024____2020_08_19_14_35_43_823739_umap_augmented",
            "full": "cifar10_0.0_full____2020_08_19_14_32_51_275942_umap_augmented",
        },
        "umap_not_augmented_thresh": {
            4: "cifar10_0.8_4____2020_08_19_23_00_09_641532_umap_augmented",  #
            16: "cifar10_0.8_16____2020_08_19_23_00_00_125286_umap_augmented",
            64: "cifar10_0.8_64____2020_08_19_23_00_54_552899_umap_augmented",
            256: "cifar10_0.8_256____2020_08_19_23_00_56_468894_umap_augmented",
            1024: "cifar10_0.8_1024____2020_08_19_23_00_59_934762_umap_augmented",
            "full": "cifar10_0.8_full____2020_08_19_23_01_03_044142_umap_augmented",
        },
        "umap_euclidean_augmented_no_thresh": {
            4: "cifar10_0.0_4____2020_08_20_10_49_23_565699_umap_augmented",  #
            16: "cifar10_0.0_16____2020_08_20_10_52_39_313456_umap_augmented",
            64: "cifar10_0.0_64____2020_08_20_10_52_40_783860_umap_augmented",
            256: "cifar10_0.0_256____2020_08_20_10_52_47_615557_umap_augmented",
            1024: "cifar10_0.0_1024____2020_08_20_10_52_58_310917_umap_augmented",
            "full": "cifar10_0.0_full____2020_08_20_10_53_00_819968_umap_augmented",
        },
    },
    "mnist": {
        "augmented": {4: "", 16: "", 64: "", 256: "", 1024: "", "full": "",},
        "not_augmented": {4: "", 16: "", 64: "", 256: "", 1024: "", "full": "",},
    },
    "fmnist": {
        "augmented": {4: "", 16: "", 64: "", 256: "", 1024: "", "full": "",},
        "not_augmented": {4: "", 16: "", 64: "", 256: "", 1024: "", "full": "",},
    },
}


def load_pretrained_weights(dataset, augmented, labels_per_class, encoder, classifier):
    aug_str = "augmented" if augmented else "not_augmented"
    pretrained_weights_loc = pretrained_networks[dataset][aug_str][labels_per_class]
    load_folder = (
        MODEL_DIR
        / "semisupervised-keras"
        / dataset
        / str(labels_per_class)
        / pretrained_weights_loc
    )
    classifier.load_weights((load_folder / "classifier").as_posix())
    encoder.load_weights((load_folder / "encoder").as_posix())
    return encoder, classifier


def load_dataset(dataset, labels_per_class):
    if dataset == "cifar10":
        X_train, X_test, X_valid, Y_train, Y_test, Y_valid = load_CIFAR10(flatten=False)
        num_classes = 10
        dims = (32, 32, 3)
    elif dataset == "mnist":
        pass
    elif dataset == "fmnist":
        pass

    # get labeled data
    if labels_per_class == "full":
        X_labeled = X_train
        Y_masked = Y_labeled = Y_train
    else:
        X_labeled, Y_labeled, Y_masked = mask_labels(
            X_train, Y_train, labels_per_class=labels_per_class
        )

    # create one hot representation
    Y_valid_one_hot = tf.keras.backend.one_hot(Y_valid, num_classes)
    Y_labeled_one_hot = tf.keras.backend.one_hot(Y_labeled, num_classes)

    return (
        X_train,
        X_test,
        X_labeled,
        Y_labeled,
        Y_masked,
        X_valid,
        Y_train,
        Y_test,
        Y_valid,
        Y_valid_one_hot,
        Y_labeled_one_hot,
        num_classes,
        dims,
    )


def load_architecture(dataset, n_latent_dims, extend_embedder=True):
    if dataset == "cifar10":
        return load_cifar10_CNN13(n_latent_dims, extend_embedder)
    elif dataset == "mnist":
        pass
    elif data == "fmnist":
        pass


from tensorflow.keras import datasets, layers, models
from tensorflow_addons.layers import WeightNormalization


def load_cifar10_CNN13(
    n_latent_dims,
    extend_embedder=True,
    dims=(32, 32, 3),
    num_classes=10,
    lr_alpha=0.1,
    dropout_rate=0.5,
):
    """
    references for network:
        - https://github.com/benathi/fastswa-semi-sup/blob/master/mean_teacher/architectures.py
        - https://github.com/vikasverma1077/ICT/blob/master/networks/lenet.py
        - https://github.com/brain-research/realistic-ssl-evaluation
    """

    def conv_block(filts, name, kernel_size=(3, 3), padding="same", **kwargs):
        return WeightNormalization(
            layers.Conv2D(
                filts, kernel_size, activation=None, padding=padding, **kwargs
            ),
            name="conv" + name,
        )

    encoder = models.Sequential()
    encoder.add(tf.keras.Input(shape=dims))
    ### conv1a
    name = "1a"
    encoder.add(conv_block(name=name, filts=128, kernel_size=(3, 3), padding="same"))
    encoder.add(layers.BatchNormalization(name="bn" + name))
    encoder.add(layers.LeakyReLU(alpha=lr_alpha, name="lrelu" + name))

    ### conv1b
    name = "1b"
    encoder.add(conv_block(name=name, filts=128, kernel_size=(3, 3), padding="same"))
    encoder.add(layers.BatchNormalization(name="bn" + name))
    encoder.add(layers.LeakyReLU(alpha=lr_alpha, name="lrelu" + name))

    ### conv1c
    name = "1c"
    encoder.add(conv_block(name=name, filts=128, kernel_size=(3, 3), padding="same"))
    encoder.add(layers.BatchNormalization(name="bn" + name))
    encoder.add(layers.LeakyReLU(alpha=lr_alpha, name="lrelu" + name))

    # max pooling
    encoder.add(
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid", name="mp1")
    )
    # dropout
    encoder.add(layers.Dropout(dropout_rate, name="drop1"))

    ### conv2a
    name = "2a"
    encoder.add(conv_block(name=name, filts=256, kernel_size=(3, 3), padding="same"))
    encoder.add(layers.BatchNormalization(name="bn" + name))
    encoder.add(layers.LeakyReLU(alpha=lr_alpha))

    ### conv2b
    name = "2b"
    encoder.add(conv_block(name=name, filts=256, kernel_size=(3, 3), padding="same"))
    encoder.add(layers.BatchNormalization(name="bn" + name))
    encoder.add(layers.LeakyReLU(alpha=lr_alpha, name="lrelu" + name))

    ### conv2c
    name = "2c"
    encoder.add(conv_block(name=name, filts=256, kernel_size=(3, 3), padding="same"))
    encoder.add(layers.BatchNormalization(name="bn" + name))
    encoder.add(layers.LeakyReLU(alpha=lr_alpha, name="lrelu" + name))

    # max pooling
    encoder.add(
        layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="valid", name="mp2")
    )
    # dropout
    encoder.add(layers.Dropout(dropout_rate, name="drop2"))

    ### conv3a
    name = "3a"
    encoder.add(conv_block(name=name, filts=512, kernel_size=(3, 3), padding="valid"))
    encoder.add(layers.BatchNormalization(name="bn" + name))
    encoder.add(layers.LeakyReLU(alpha=lr_alpha, name="lrelu" + name))

    ### conv3b
    name = "3b"
    encoder.add(conv_block(name=name, filts=256, kernel_size=(1, 1), padding="valid"))
    encoder.add(layers.BatchNormalization(name="bn" + name))
    encoder.add(layers.LeakyReLU(alpha=lr_alpha, name="lrelu" + name))

    ### conv3c
    name = "3c"
    encoder.add(conv_block(name=name, filts=128, kernel_size=(1, 1), padding="valid"))
    encoder.add(layers.BatchNormalization(name="bn" + name))
    encoder.add(layers.LeakyReLU(alpha=lr_alpha, name="lrelu" + name))

    # max pooling
    encoder.add(layers.AveragePooling2D(pool_size=(6, 6), strides=2, padding="valid"))
    encoder.add(layers.Flatten())
    encoder.add(layers.Dense(256, activation=None, name="z"))

    classifier = models.Sequential()
    classifier.add(tf.keras.Input(shape=(256)))
    classifier.add(WeightNormalization(layers.Dense(256, activation=None)))
    classifier.add(layers.LeakyReLU(alpha=lr_alpha, name="lrelufc1"))
    classifier.add(WeightNormalization(layers.Dense(256, activation=None)))
    classifier.add(layers.LeakyReLU(alpha=lr_alpha, name="lrelufc2"))
    classifier.add(
        WeightNormalization(layers.Dense(num_classes, activation=None), name="y_")
    )

    embedder = models.Sequential()
    embedder.add(tf.keras.Input(shape=(256)))
    if extend_embedder:
        embedder.add(WeightNormalization(layers.Dense(256, activation=None)))
        embedder.add(layers.LeakyReLU(alpha=lr_alpha, name="lrelufc1"))
        embedder.add(WeightNormalization(layers.Dense(256, activation=None)))
        embedder.add(layers.LeakyReLU(alpha=lr_alpha, name="lrelufc2"))
    embedder.add(
        WeightNormalization(layers.Dense(n_latent_dims, activation=None), name="z_")
    )

    return encoder, classifier, embedder


from scipy import optimize


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


def convert_distance_to_probability(distances, a, b):
    """ convert distance representation into probability, 
        as a function of a, b params
    """
    return 1.0 / (1.0 + a * distances ** (2 * b))


def compute_cross_entropy(
    probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0
):
    """
    Compute cross entropy between low and high probability

    Parameters
    ----------
    probabilities_graph : [type]
        high dimensional probabilities
    probabilities_distance : [type]
        low dimensional probabilities
    EPS : [type], optional
        offset to to ensure log is taken of a positive number, by default 1e-4
    repulsion_strength : float, optional
        strength of repulsion between negative samples, by default 1.0

    Returns
    -------
    attraction_term: tf.float32
        attraction term for cross entropy loss
    repellant_term: tf.float32
        repellant term for cross entropy loss
    cross_entropy: tf.float32
        cross entropy umap loss
    
    """
    # cross entropy
    attraction_term = -probabilities_graph * tf.math.log(
        tf.clip_by_value(probabilities_distance, EPS, 1.0)
    )
    repellant_term = (
        -(1.0 - probabilities_graph)
        * tf.math.log(tf.clip_by_value(1.0 - probabilities_distance, EPS, 1.0))
        * repulsion_strength
    )

    # balance the expected losses between atrraction and repel
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE


def umap_loss(batch_size, negative_sample_rate, _a, _b, repulsion_strength=1.0):
    @tf.function
    def loss(placeholder_y, embed_to_from):
        # split out to/from
        embedding_to, embedding_from = tf.split(
            embed_to_from, num_or_size_splits=2, axis=1
        )

        # get negative samples
        embedding_neg_to = tf.repeat(embedding_to, negative_sample_rate, axis=0)
        repeat_neg = tf.repeat(embedding_from, negative_sample_rate, axis=0)
        embedding_neg_from = tf.gather(
            repeat_neg, tf.random.shuffle(tf.range(tf.shape(repeat_neg)[0]))
        )

        #  distances between samples (and negative samples)
        distance_embedding = tf.concat(
            [
                tf.norm(embedding_to - embedding_from, axis=1),
                tf.norm(embedding_neg_to - embedding_neg_from, axis=1),
            ],
            axis=0,
        )

        # convert probabilities to distances
        probabilities_distance = convert_distance_to_probability(
            distance_embedding, _a, _b
        )

        # set true probabilities based on negative sampling
        probabilities_graph = tf.concat(
            [tf.ones(batch_size), tf.zeros(batch_size * negative_sample_rate)], axis=0,
        )

        # compute cross entropy
        (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=repulsion_strength,
        )
        return tf.reduce_mean(ce_loss)

    return loss


def build_model(
    batch_size,
    a_param,
    b_param,
    dims,
    embedder,
    encoder,
    classifier,
    negative_sample_rate=5,
    optimizer=tf.keras.optimizers.Adam(1e-3),
    label_smoothing=0.2,
    umap_weight=1.0,
):

    # inputs
    to_x = tf.keras.layers.Input(shape=dims, name="to_x")
    from_x = tf.keras.layers.Input(shape=dims, name="from_x")
    classifier_x = tf.keras.layers.Input(shape=dims, name="classifier_x")

    # embeddings
    if embedder is not None:
        embedding_to = embedder(encoder(to_x))
        embedding_from = embedder(encoder(from_x))
    else:
        embedding_to = encoder(to_x)
        embedding_from = encoder(from_x)
    embedding_to_from = tf.concat([embedding_to, embedding_from], axis=1)
    embedding_to_from = tf.keras.layers.Lambda(lambda x: x, name="umap")(
        embedding_to_from
    )

    # predictions
    predictions = classifier(encoder(classifier_x))
    predictions = tf.keras.layers.Lambda(lambda x: x, name="classifier")(predictions)

    # create model
    model = tf.keras.Model(
        inputs=[classifier_x, to_x, from_x],
        outputs={"classifier": predictions, "umap": embedding_to_from},
    )

    # compile model
    model.compile(
        optimizer=optimizer,
        loss={
            "classifier": tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=label_smoothing, from_logits=True
            ),
            "umap": umap_loss(batch_size, negative_sample_rate, a_param, b_param),
        },
        loss_weights={"classifier": 1.0, "umap": umap_weight},
        metrics={"classifier": "accuracy"},
    )

    return model


import tensorflow_addons as tfa


def get_augment(dims):
    def augment(image):
        image = tf.squeeze(image)  # Add 6 pixels of padding
        image = tf.image.resize_with_crop_or_pad(
            image, dims[0] + 6, dims[1] + 6
        )  # crop 6 pixels
        image = tf.image.random_crop(image, size=dims)
        image = tf.image.random_brightness(image, max_delta=0.15)  # Random brightness
        image = tf.image.random_contrast(image, lower=0.6, upper=1.4)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_hue(image, 0.05, seed=None)
        # image = tf.image.random_jpeg_quality(image, 10, 100, seed=None)
        image = tfa.image.rotate(
            image,
            tf.squeeze(tf.random.uniform(shape=(1, 1), minval=-0.2, maxval=0.2)),
            interpolation="BILINEAR",
        )
        image = tfa.image.random_cutout(
            tf.expand_dims(image, 0), (8, 8), constant_values=0.5
        )[0]
        image = tf.clip_by_value(image, 0, 1)
        return image

    return augment


def build_labeled_iterator(X_labeled, Y_labeled_one_hot, augmented, dims):
    def aug_labeled(image, label):
        return get_augment(dims)(image), label

    labeled_dataset = tf.data.Dataset.from_tensor_slices((X_labeled, Y_labeled_one_hot))
    labeled_dataset = labeled_dataset.shuffle(len(X_labeled))
    labeled_dataset = labeled_dataset.repeat()
    if augmented:
        labeled_dataset = labeled_dataset.map(
            aug_labeled, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    return labeled_dataset


def rearrange_output(label_iter_out, edge_iter_out):
    X_lab, Y_lab = label_iter_out
    (X_to, X_from), edge_prob = edge_iter_out
    return (X_lab, X_to, X_from), {"classifier": Y_lab, "umap": edge_prob}


from pynndescent import NNDescent
from umap.umap_ import fuzzy_simplicial_set, discrete_metric_simplicial_set_intersection
from sklearn.utils import check_random_state, check_array


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


def prepare_edge_dataset(
    data,
    augmented,
    batch_size,
    labeled_dataset,
    X_train_hc,
    Y_masked,
    dims,
    n_neighbors=15,
    n_epochs=200,
    max_sample_repeats_per_epoch=25,
):
    def gather_X(edge_to, edge_from):
        return (tf.gather(X_train_hc, edge_to), tf.gather(X_train_hc, edge_from)), 0

    def aug_edge(images, edge_weight):
        return (augment(images[0]), augment(images[1])), edge_weight

    augment = get_augment(dims)

    # flatten if needed
    if len(np.shape(data)) > 2:
        data = data.reshape((len(data), np.product(np.shape(data)[1:])))

    # compute umap graph
    umap_graph = build_fuzzy_simplicial_set(data, y=Y_masked, n_neighbors=n_neighbors,)

    # get graph elements
    graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(
        umap_graph, n_epochs
    )

    # set the maximum number of times each edge should be repeated per epoch
    epochs_per_sample = np.clip(
        (weight / np.max(weight)) * max_sample_repeats_per_epoch,
        1,
        max_sample_repeats_per_epoch,
    ).astype("int")

    # repeat based on epochs per sample
    edges_to_exp, edges_from_exp = (
        np.repeat(head, epochs_per_sample.astype("int")),
        np.repeat(tail, epochs_per_sample.astype("int")),
    )

    # shuffle everything
    shuffle_mask = np.random.permutation(range(len(edges_to_exp)))
    edges_to_exp = edges_to_exp[shuffle_mask]
    edges_from_exp = edges_from_exp[shuffle_mask]

    # create iterator
    edge_iter = tf.data.Dataset.from_tensor_slices((edges_to_exp, edges_from_exp))
    edge_iter = edge_iter.shuffle(10000)
    edge_iter = edge_iter.repeat()
    edge_iter = edge_iter.map(
        gather_X, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    if augmented:
        edge_iter = edge_iter.map(
            aug_edge, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

    return edge_iter


def flatten(x):
    shape_x = np.shape(x)
    return np.reshape(x, [len(x)] + list(np.shape(x)[1:]))


def get_edge_dataset(
    model,
    augmented,
    classifier,
    encoder,
    X_train,
    Y_masked,
    batch_size,
    confidence_threshold,
    labeled_dataset,
    dims,
    learned_metric,
):

    if learned_metric:

        # model to grab last layer activations
        last_layer_class = tf.keras.models.Model(
            classifier.input,
            [classifier.get_layer(name=classifier.layers[-2].name).get_output_at(0)],
        )

        # get encoder activations for X_train
        enc_z = encoder.predict(X_train)

        # get last layer activations for X_train
        last_layer_z = last_layer_class.predict(enc_z)

        # get predictions for X_train
        train_predictions = classifier.predict(
            encoder.predict(X_train, batch_size=128, verbose=True),
            batch_size=128,
            verbose=True,
        )
        pred_softmax = tf.nn.softmax(train_predictions, axis=1)

        # confidence for predictions
        confidence = np.max(pred_softmax, axis=1)

        #
        high_confidence_mask = confidence > confidence_threshold
        X_train_hc = X_train[high_confidence_mask]

        edge_dataset = prepare_edge_dataset(
            last_layer_z[high_confidence_mask],
            augmented,
            batch_size,
            labeled_dataset,
            X_train_hc,
            Y_masked,
            dims,
        )

    else:

        edge_dataset = prepare_edge_dataset(
            flatten(X_train),
            augmented,
            batch_size,
            labeled_dataset,
            X_train,
            Y_masked,
            dims,
        )

    return edge_dataset


from umap.umap_ import make_epochs_per_sample


def get_graph_elements(graph_, n_epochs):
    """
    gets elements of graphs, weights, and number of epochs per edge

    Parameters
    ----------
    graph_ : [type]
        umap graph of probabilities
    n_epochs : int
        maximum number of epochs per edge

    Returns
    -------
    graph [type]
        umap graph
    epochs_per_sample np.array
        number of epochs to train each sample for
    head np.array
        edge head
    tail np.array
        edge tail
    weight np.array
        edge weight
    n_vertices int
        number of verticies in graph
    """
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


def zip_datasets(labeled_dataset, edge_dataset, batch_size):
    # merge with  labeled iterator
    zipped_ds = tf.data.Dataset.zip((labeled_dataset, edge_dataset))
    zipped_ds = zipped_ds.map(
        rearrange_output, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    zipped_ds = zipped_ds.batch(batch_size, drop_remainder=True)
    return zipped_ds

from tfumap.load_datasets import (
    load_CIFAR10,
    load_MNIST,
    load_FMNIST,
    load_CASSINS,
    load_MACOSKO,
    mask_labels,
)
import tensorflow as tf
from tfumap.paths import MODEL_DIR
import numpy as np


pretrained_networks = {
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
        "umap_augmented_learned": {
            4: "cifar10_0.0_4____2020_08_19_00_40_15_425455_umap_augmented",  #
            16: "cifar10_0.0_16____2020_08_19_00_40_13_037112_umap_augmented",
            64: "cifar10_0.0_64____2020_08_19_00_40_13_032397_umap_augmented",
            256: "cifar10_0.0_256____2020_08_18_16_16_47_694512_umap_augmented",
            1024: "cifar10_0.0_1024____2020_08_19_10_25_26_973224_umap_augmented",
            "full": "cifar10_0.0_full____2020_08_19_00_40_18_936212_umap_augmented",
        },
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
        "umap_euclidean_augmented": {  # umap_euclidean_augmented_no_thresh
            4: "cifar10_0.0_4____2020_08_20_10_49_23_565699_umap_augmented",  #
            16: "cifar10_0.0_16____2020_08_20_10_52_39_313456_umap_augmented",
            64: "cifar10_0.0_64____2020_08_20_10_52_40_783860_umap_augmented",
            256: "cifar10_0.0_256____2020_08_20_10_52_47_615557_umap_augmented",
            1024: "cifar10_0.0_1024____2020_08_20_10_52_58_310917_umap_augmented",
            "full": "cifar10_0.0_full____2020_08_20_10_53_00_819968_umap_augmented",
        },
        "umap_not_augmented_linear_thresh": {
            4: "cifar10_0.8_4____2020_08_22_22_47_43_598023_umap_augmented",  #
            16: "cifar10_0.8_16____2020_08_22_22_47_57_967494_umap_augmented",
            64: "cifar10_0.8_64____2020_08_22_22_47_27_952365_umap_augmented",
            256: "cifar10_0.8_256____2020_08_22_22_48_38_890043_umap_augmented",
            1024: "cifar10_0.8_1024____2020_08_22_22_49_43_660652_umap_augmented",
            "full": "cifar10_0.8_full____2020_08_22_22_49_37_683086_umap_augmented",
        },
        "umap_euclidean": {
            4: "cifar10_0.0_4____2020_08_24_10_10_03_033874_umap_augmented",
            16: "cifar10_0.0_16____2020_08_24_00_26_41_868150_umap_augmented",
            64: "cifar10_0.0_64____2020_08_24_00_26_53_791994_umap_augmented",
            256: "cifar10_0.0_256____2020_08_24_00_22_53_202346_umap_augmented",
            1024: "cifar10_0.0_1024____2020_08_24_00_22_53_212673_umap_augmented",
            "full": "cifar10_0.0_full____2020_08_23_23_52_33_359986_umap_augmented",
        },
        "umap_learned": {
            4: "cifar10_0.0_4____2020_08_19_14_35_43_127626_umap_augmented",  #
            16: "cifar10_0.0_16____2020_08_19_14_35_43_867336_umap_augmented",
            64: "cifar10_0.0_64____2020_08_19_14_35_43_736036_umap_augmented",
            256: "cifar10_0.0_256____2020_08_19_14_38_31_105228_umap_augmented",
            1024: "cifar10_0.0_1024____2020_08_19_14_35_43_823739_umap_augmented",
            "full": "cifar10_0.0_full____2020_08_19_14_32_51_275942_umap_augmented",
        },
    },
    "mnist": {
        "not_augmented": {
            4: "mnist_4____2020_08_23_13_59_31_357892_baseline",
            16: "mnist_16____2020_08_23_14_13_03_306576_baseline",
            64: "mnist_64____2020_08_23_14_13_19_397319_baseline",
            256: "mnist_256____2020_08_23_14_12_28_828611_baseline",
            1024: "mnist_1024____2020_08_23_14_12_00_839816_baseline",
            "full": "mnist_full____2020_08_23_14_02_35_917340_baseline",
        },
        "augmented": {
            4: "mnist_4____2020_08_26_22_34_26_172040_baseline_augmented",
            16: "mnist_16____2020_08_26_22_36_42_823740_baseline_augmented",
            64: "mnist_64____2020_08_26_22_37_03_013806_baseline_augmented",
            256: "mnist_256____2020_08_26_22_38_00_695064_baseline_augmented",
            1024: "mnist_1024____2020_08_26_22_38_22_879325_baseline_augmented",
            "full": "mnist_full____2020_08_26_22_34_57_589833_baseline_augmented",
        },
        "umap_euclidean": {
            4: "mnist_0.0_4____2020_08_23_19_39_30_768509_umap_augmented",
            16: "mnist_0.0_16____2020_08_23_19_27_31_722774_umap_augmented",
            64: "mnist_0.0_64____2020_08_23_18_32_38_592348_umap_augmented",
            256: "mnist_0.0_256____2020_08_23_19_39_57_288829_umap_augmented",
            1024: "mnist_0.0_1024____2020_08_23_19_44_01_747431_umap_augmented",
            # "full": "mnist_0.0_full____2020_08_23_23_07_06_598185_umap_augmented",
            "full": "mnist_0.0_full____2020_08_23_23_11_15_364937_umap_augmented",
        },
        "umap_learned": {
            4: "mnist_0.0_4____2020_08_24_13_43_38_697668_umap_augmented",
            16: "mnist_0.0_16____2020_08_24_16_51_10_703116_umap_augmented",
            64: "mnist_0.0_64____2020_08_24_16_51_16_969542_umap_augmented",
            256: "mnist_0.0_256____2020_08_24_16_53_11_404946_umap_augmented",
            1024: "mnist_0.0_1024____2020_08_24_16_53_15_376183_umap_augmented",
            "full": "mnist_0.0_full____2020_08_24_13_43_38_497837_umap_augmented",
        },
        "umap_euclidean_augmented": {
            4: "mnist_0.0_4____2020_08_28_01_19_15_530909_umap_augmented",
            16: "mnist_0.0_16____2020_08_28_01_22_00_266602_umap_augmented",
            64: "mnist_0.0_64____2020_08_28_01_22_22_251679_umap_augmented",
            256: "mnist_0.0_256____2020_08_28_01_22_37_322969_umap_augmented",
            1024: "mnist_0.0_1024____2020_08_28_10_21_10_652408_umap_augmented",
            "full": "mnist_0.0_full____2020_08_28_10_21_10_309737_umap_augmented",
        },
        "umap_augmented_learned": {
            4: "mnist_0.0_4____2020_08_28_22_36_55_334573_umap_augmented",
            16: "mnist_0.0_16____2020_08_28_22_36_36_245588_umap_augmented",
            64: "mnist_0.0_64____2020_08_28_10_58_23_001653_umap_augmented",
            256: "mnist_0.0_256____2020_08_28_10_58_44_499275_umap_augmented",
            1024: "mnist_0.0_1024____2020_08_28_11_00_20_544491_umap_augmented",
            "full": "mnist_0.0_full____2020_08_28_11_00_23_221668_umap_augmented",
        },
    },
    "fmnist": {
        "not_augmented": {
            4: "fmnist_4____2020_08_23_14_15_38_194490_baseline",
            16: "fmnist_16____2020_08_23_14_15_50_074976_baseline",
            64: "fmnist_64____2020_08_23_14_16_00_145880_baseline",
            256: "fmnist_256____2020_08_23_14_14_27_904250_baseline",
            1024: "fmnist_1024____2020_08_23_14_13_39_538728_baseline",
            "full": "fmnist_full____2020_08_23_14_06_13_546999_baseline",
        },
        "augmented": {
            4: "fmnist_4____2020_08_25_17_18_57_856259_baseline_augmented",
            16: "fmnist_16____2020_08_25_17_19_58_221943_baseline_augmented",
            64: "fmnist_64____2020_08_25_17_20_33_647542_baseline_augmented",
            256: "fmnist_256____2020_08_25_17_20_55_354044_baseline_augmented",
            1024: "fmnist_1024____2020_08_25_17_21_21_486291_baseline_augmented",
            "full": "fmnist_full____2020_08_25_17_21_42_014099_baseline_augmented",
        },
        "placeholder": {4: "", 16: "", 64: "", 256: "", 1024: "", "full": ""},
        "umap_over_z": {
            4: "fmnist_0.0_4____2020_08_27_11_30_38_602000_umap_augmented",
            16: "fmnist_0.0_16____2020_08_27_11_30_41_024752_umap_augmented",
            64: "fmnist_0.0_64____2020_08_27_11_46_44_906423_umap_augmented",
            256: "fmnist_0.0_256____2020_08_27_11_47_02_912498_umap_augmented",
            1024: "",
            "full": "",
        },
        "umap_augmented_learned": {
            4: "fmnist_0.0_4____2020_08_25_22_52_13_661088_umap_augmented",
            16: "fmnist_0.0_16____2020_08_25_22_53_12_075808_umap_augmented",
            64: "fmnist_0.0_64____2020_08_25_22_58_52_822672_umap_augmented",
            256: "fmnist_0.0_256____2020_08_25_22_59_00_936495_umap_augmented",
            1024: "fmnist_0.0_1024____2020_08_25_22_59_15_453823_umap_augmented",
            "full": "fmnist_0.0_full____2020_08_25_22_59_11_829778_umap_augmented",
        },
        "umap_euclidean": {
            4: "fmnist_0.0_4____2020_08_23_18_48_03_409056_umap_augmented",
            16: "fmnist_0.0_16____2020_08_23_21_25_30_890380_umap_augmented",
            64: "fmnist_0.0_64____2020_08_23_19_43_20_063919_umap_augmented",
            256: "fmnist_0.0_256____2020_08_23_19_44_36_506473_umap_augmented",
            1024: "fmnist_0.0_1024____2020_08_23_21_25_43_287069_umap_augmented",
            "full": "fmnist_0.0_full____2020_08_23_23_13_31_899132_umap_augmented",
        },
        "umap_learned": {
            4: "fmnist_0.0_4____2020_08_24_10_19_02_171374_umap_augmented",
            16: "fmnist_0.0_16____2020_08_24_10_19_11_697170_umap_augmented",
            64: "fmnist_0.0_64____2020_08_24_10_19_33_327157_umap_augmented",
            256: "fmnist_0.0_256____2020_08_24_10_19_51_978912_umap_augmented",
            1024: "fmnist_0.0_1024____2020_08_24_10_20_06_630456_umap_augmented",
            "full": "fmnist_0.0_full____2020_08_24_10_20_11_972145_umap_augmented",
        },
        "umap_intersection": {
            4: "fmnist_0.0_4____2020_08_24_23_43_25_574078_umap_augmented",
            16: "fmnist_0.0_16____2020_08_24_23_43_35_567328_umap_augmented",
            64: "fmnist_0.0_64____2020_08_24_23_43_35_567450_umap_augmented",
            256: "fmnist_0.0_256____2020_08_24_23_43_45_557361_umap_augmented",
            1024: "fmnist_0.0_1024____2020_08_24_23_43_45_643845_umap_augmented",
            "full": "fmnist_0.0_full____2020_08_24_23_48_54_578235_umap_augmented",
        },
        "umap_euclidean_augmented": {
            4: "fmnist_0.0_4____2020_08_26_11_16_46_042019_umap_augmented",
            16: "fmnist_0.0_16____2020_08_26_13_30_25_749568_umap_augmented",
            64: "fmnist_0.0_64____2020_08_26_13_30_25_380156_umap_augmented",
            256: "fmnist_0.0_256____2020_08_26_11_21_26_903869_umap_augmented",
            1024: "fmnist_0.0_1024____2020_08_26_11_21_26_883542_umap_augmented",
            "full": "fmnist_0.0_full____2020_08_26_13_30_25_074505_umap_augmented",
        },
    },
    "cassins": {
        "not_augmented": {
            4: "cassins_4____2020_11_11_17_40_13_549097_baseline",
            16: "cassins_16____2020_11_11_17_56_52_163428_baseline",
            64: "cassins_64____2020_11_11_17_56_57_053996_baseline",
            256: "cassins_256____2020_11_11_18_03_06_161985_baseline",
            1024: "cassins_1024____2020_11_11_17_59_57_592404_baseline",
            "full": "cassins_full____2020_11_11_17_40_24_970030_baseline",
        },
        "umap_euclidean": {
            4: "cassins_0.0_4____2020_11_11_19_13_44_404196_umap_augmented",
            16: "cassins_0.0_16____2020_11_11_19_15_56_651149_umap_augmented",
            64: "cassins_0.0_64____2020_11_11_19_16_36_796314_umap_augmented",
            256: "cassins_0.0_256____2020_11_11_19_17_09_334810_umap_augmented",
            1024: "cassins_0.0_1024____2020_11_11_19_17_44_566481_umap_augmented",
            "full": "cassins_0.0_full____2020_11_11_19_18_52_525170_umap_augmented",
        },
    },
    "macosko2015": {
        "not_augmented": {
            4: "macosko2015_4____2020_11_11_20_08_20_373036_baseline",
            16: "macosko2015_16____2020_11_11_19_41_43_082763_baseline",
            64: "macosko2015_64____2020_11_11_19_46_03_845826_baseline",
            256: "macosko2015_256____2020_11_11_19_46_22_012555_baseline",
            1024: "macosko2015_1024____2020_11_11_19_46_40_990058_baseline",
            "full": "macosko2015_full____2020_11_11_19_47_00_582248_baseline",
        },
        "umap_euclidean": {
            4: "macosko2015_0.0_4____2020_11_11_21_10_18_248580_umap_augmented",
            16: "macosko2015_0.0_16____2020_11_11_20_07_36_383558_umap_augmented",
            64: "macosko2015_0.0_64____2020_11_11_20_12_03_040798_umap_augmented",
            256: "macosko2015_0.0_256____2020_11_11_20_13_05_560591_umap_augmented",
            1024: "macosko2015_0.0_1024____2020_11_11_20_13_34_650588_umap_augmented",
            "full": "macosko2015_0.0_full____2020_11_11_20_13_53_727456_umap_augmented",
        },
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
        X_train, X_test, X_valid, Y_train, Y_test, Y_valid = load_MNIST(flatten=False)
        num_classes = 10
        dims = (28, 28, 1)
    elif dataset == "fmnist":
        X_train, X_test, X_valid, Y_train, Y_test, Y_valid = load_FMNIST(flatten=False)
        num_classes = 10
        dims = (28, 28, 1)
    elif dataset == "cassins":
        X_train, X_test, X_valid, Y_train, Y_test, Y_valid = load_CASSINS(flatten=False)
        num_classes = 20
        dims = (32, 31, 1)
    elif dataset == "macosko2015":
        X_train, X_test, X_valid, Y_train, Y_test, Y_valid = load_MACOSKO(flatten=False)
        num_classes = 12
        dims = 50

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
        return load_mnist_CNN(n_latent_dims, extend_embedder)
    elif dataset == "fmnist":
        return load_mnist_CNN(n_latent_dims, extend_embedder)
    elif dataset == "cassins":
        return load_cassins_RNN(n_latent_dims, extend_embedder)
    elif dataset == "macosko2015":
        return load_macosko_NET(n_latent_dims, extend_embedder)


from tensorflow.keras import datasets, layers, models
from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.layers import (
    Conv2D,
    Reshape,
    Bidirectional,
    Dense,
    RepeatVector,
    TimeDistributed,
    LSTM,
)


def load_macosko_NET(
    n_latent_dims,
    extend_embedder=True,
    dims=(50),
    num_classes=12,
    lr_alpha=0.1,
    dropout_rate=0.5,
):
    """

    """

    encoder = models.Sequential()
    encoder.add(tf.keras.Input(shape=dims))
    encoder.add(layers.Dense(256, activation=None))
    encoder.add(layers.LeakyReLU(alpha=lr_alpha))
    encoder.add(layers.Dense(256, activation=None))
    encoder.add(layers.LeakyReLU(alpha=lr_alpha))
    encoder.add(layers.Dense(256, activation=None))
    encoder.add(layers.LeakyReLU(alpha=lr_alpha))
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


def load_cassins_RNN(
    n_latent_dims,
    extend_embedder=True,
    dims=(32, 31, 1),
    num_classes=20,
    lr_alpha=0.1,
    dropout_rate=0.5,
):
    """

    """

    encoder = models.Sequential()
    encoder.add(tf.keras.Input(shape=dims))
    encoder.add(
        Conv2D(
            filters=32,
            kernel_size=3,
            strides=(2, 2),
            activation=tf.nn.leaky_relu,
            padding="same",
        )
    )
    encoder.add(
        Conv2D(
            filters=64,
            kernel_size=3,
            strides=(2, 2),
            activation=tf.nn.leaky_relu,
            padding="same",
        )
    )
    encoder.add(
        Conv2D(
            filters=128,
            kernel_size=3,
            strides=(2, 1),
            activation=tf.nn.leaky_relu,
            padding="same",
        )
    )
    encoder.add(
        Conv2D(
            filters=128,
            kernel_size=3,
            strides=(2, 1),
            activation=tf.nn.leaky_relu,
            padding="same",
        )
    )
    encoder.add(Reshape(target_shape=(8, 2 * 128)))
    encoder.add(Bidirectional(LSTM(units=100, activation="relu")))
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


def load_mnist_CNN(
    n_latent_dims,
    extend_embedder=True,
    dims=(28, 28, 1),
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
    encoder.add(layers.AveragePooling2D(pool_size=(3, 3), strides=2, padding="valid"))
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


def get_augment_cifar(dims):
    def augment(image):
        # image = tf.squeeze(image)  # Add 6 pixels of padding
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


def get_augment_fmnist(dims):
    def augment(image):
        def random_stretch(image):
            # stretch
            randint_hor = tf.random.uniform((2,), minval=0, maxval=8, dtype=tf.int32)[0]
            randint_vert = tf.random.uniform((2,), minval=0, maxval=8, dtype=tf.int32)[
                0
            ]
            image = tf.image.resize(
                image, (dims[0] + randint_vert * 2, dims[1] + randint_hor * 2)
            )
            # image = tf.image.crop_to_bounding_box(image, randint_vert,randint_hor,28,28)
            image = tf.image.resize_with_pad(image, dims[0], dims[1])

            image = tf.image.resize_with_crop_or_pad(
                image, dims[0] + 3, dims[1] + 3
            )  # crop 6 pixels
            image = tf.image.random_crop(image, size=dims)

            return image

        random_switch = tf.cast(
            tf.random.uniform((1,), minval=0, maxval=2, dtype=tf.int32)[0], tf.bool
        )
        image = tf.cond(random_switch, lambda: random_stretch(image), lambda: image)

        def rotate_image(image):
            image = tfa.image.rotate(
                image,
                tf.squeeze(tf.random.uniform(shape=(1, 1), minval=-0.25, maxval=0.25)),
                interpolation="BILINEAR",
            )
            return image

        random_switch = tf.cast(
            tf.random.uniform((1,), minval=0, maxval=2, dtype=tf.int32)[0], tf.bool
        )
        image = tf.cond(random_switch, lambda: rotate_image(image), lambda: image)

        image = tf.image.random_flip_left_right(image)
        image = tf.clip_by_value(image, 0, 1)

        def norm(x):
            return x - tf.reduce_min(x)  # /(tf.reduce_max(x) - tf.reduce_min(x))

        def adjust_hue_brightness(image):
            image = tf.image.random_brightness(
                image, max_delta=0.5
            )  # Random brightness
            image = tf.image.random_contrast(image, lower=0.5, upper=1.75)
            image = norm(image)
            return image

        random_switch = tf.cast(
            tf.random.uniform((1,), minval=0, maxval=2, dtype=tf.int32)[0], tf.bool
        )
        image = tf.cond(
            random_switch, lambda: adjust_hue_brightness(image), lambda: image
        )

        def cutout(image):
            image = tfa.image.random_cutout(
                tf.expand_dims(image, 0), (8, 8), constant_values=0.5
            )[0]
            image = tf.clip_by_value(image, 0, 1)
            return image

        random_switch = tf.cast(
            tf.random.uniform((1,), minval=0, maxval=2, dtype=tf.int32)[0], tf.bool
        )
        image = tf.cond(random_switch, lambda: cutout(image), lambda: image)

        image = tf.clip_by_value(image, 0, 1)
        return image

    return augment


def get_augment_mnist(
    dims=(28, 28, 1),
    augment_probability=0.1,
    brightness_range=[0.5, 1],
    contrast_range=[0.5, 2],
    cutout_range=[0, 0.75],
    rescale_range=[0.75, 1],
    rescale_range_x_range=0.9,
    rescale_range_y_range=0.9,
    rotate_range=[-0.5, 0.5],
    shear_x_range=[-0.3, 0.3],
    shear_y_range=[-0.3, 0.3],
    translate_x_range=0.2,
    translate_y_range=0.2,
):
    def augment(image):
        def aug(image):
            # Brightness 0-1
            brightness_factor = tf.random.uniform(
                (1,),
                minval=brightness_range[0],
                maxval=brightness_range[1],
                dtype=tf.float32,
            )[0]
            image = brightness(image, brightness_factor)

            # rescale 0.5-1
            rescale_factor = tf.random.uniform(
                (1,), minval=rescale_range[0], maxval=rescale_range[1], dtype=tf.float32
            )[0]
            image = tf.image.random_crop(
                image, [dims[0] * rescale_factor, dims[1] * rescale_factor, dims[2]]
            )
            image = tf.image.resize(image, [dims[0], dims[1]])

            # sqeeze x or y
            randint_hor = tf.random.uniform(
                (2,),
                minval=0,
                maxval=tf.cast(rescale_range_x_range * dims[0], tf.int32),
                dtype=tf.int32,
            )[0]
            randint_vert = tf.random.uniform(
                (2,),
                minval=0,
                maxval=tf.cast(rescale_range_y_range * dims[1], tf.int32),
                dtype=tf.int32,
            )[0]
            image = tf.image.resize(
                image, (dims[0] + randint_vert * 2, dims[1] + randint_hor * 2)
            )

            image = tf.image.resize_with_pad(image, dims[0], dims[1])

            image = tf.image.resize_with_crop_or_pad(
                image, dims[0] + 3, dims[1] + 3
            )  # crop 6 pixels
            image = tf.image.random_crop(image, size=dims)

            # rotate -45 45
            rotate_factor = tf.random.uniform(
                (1,), minval=rotate_range[0], maxval=rotate_range[1], dtype=tf.float32,
            )[0]
            image = tfa.image.rotate(image, rotate_factor, interpolation="BILINEAR",)

            # shear_x -0.3, 3
            shear_x_factor = tf.random.uniform(
                (1,), minval=shear_x_range[0], maxval=shear_x_range[1], dtype=tf.float32
            )[0]

            img = tf.repeat(tf.cast(image * 255, tf.uint8), 3, axis=2)
            image = (
                tf.cast(
                    tfa.image.shear_x(img, shear_x_factor, replace=0)[:, :, :1],
                    tf.float32,
                )
                / 255
            )

            # shear_y -0.3, 3
            shear_y_factor = tf.random.uniform(
                (1,), minval=shear_x_range[0], maxval=shear_y_range[1], dtype=tf.float32
            )[0]
            img = tf.repeat(tf.cast(image * 255, tf.uint8), 3, axis=2)
            image = (
                tf.cast(
                    tfa.image.shear_y(img, shear_y_factor, replace=0)[:, :, :1],
                    tf.float32,
                )
                / 255.0
            )
            # print(image.shape)
            # translate x -0.3, 0.3
            translate_x_factor = tf.random.uniform(
                (1,), minval=0, maxval=translate_x_range * 2, dtype=tf.float32
            )[0]
            # translate y -0.3, 0.3
            translate_y_factor = tf.random.uniform(
                (1,), minval=0, maxval=translate_y_range * 2, dtype=tf.float32
            )[0]

            image = tf.image.resize_with_crop_or_pad(
                image,
                dims[0] + tf.cast(translate_x_factor * dims[0], tf.int32),
                dims[1] + tf.cast(translate_y_factor * dims[1], tf.int32),
            )  # crop 6 pixels
            image = tf.image.random_crop(image, size=dims)

            # contrast 0-1
            contrast_factor = tf.random.uniform(
                (1,),
                minval=contrast_range[0],
                maxval=contrast_range[1],
                dtype=tf.float32,
            )[0]
            image = tf.image.adjust_contrast(image, contrast_factor)
            image = image - tf.reduce_min(image)

            # cutout 0-0.5
            cutout_factor = tf.random.uniform(
                (1,), minval=cutout_range[0], maxval=cutout_range[1], dtype=tf.float32
            )[0]
            image = cutout(image, tf.cast(cutout_factor * dims[0], tf.int32))

            image = tf.clip_by_value(image, 0.0, 1.0)

            return image

        random_switch = tf.math.equal(
            tf.random.uniform(
                (),
                minval=0,
                maxval=tf.cast(1 + 1 / augment_probability, tf.int32),
                dtype=tf.int32,
            ),
            1,
        )
        return tf.cond(random_switch, lambda: aug(image), lambda: image)

    return augment


def build_labeled_iterator(
    X_labeled, Y_labeled_one_hot, augmented, dims, dataset="cifar10"
):
    if augmented:
        if dataset == "cifar10":

            def aug_labeled(image, label):
                return get_augment_cifar(dims)(image), label

        elif dataset == "fmnist":

            def aug_labeled(image, label):
                return get_augment_fmnist(dims)(image), label

        elif dataset == "mnist":

            def aug_labeled(image, label):
                return get_augment_mnist(dims)(image), label

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
    umap_graph,
    augmented,
    batch_size,
    labeled_dataset,
    X_train_hc,
    Y_masked,
    dims,
    n_neighbors=15,
    n_epochs=200,
    max_sample_repeats_per_epoch=25,
    dataset="cifar10",
):
    def gather_X(edge_to, edge_from):
        return (tf.gather(X_train_hc, edge_to), tf.gather(X_train_hc, edge_from)), 0

    def aug_edge(images, edge_weight):
        return (augment(images[0]), augment(images[1])), edge_weight

    if dataset == "cifar10":
        augment = get_augment_cifar(dims)
    elif dataset == "fmnist":
        augment = get_augment_fmnist(dims)
    elif dataset == "mnist":
        augment = get_augment_mnist(dims)

    """# flatten if needed
    if len(np.shape(data)) > 2:
        data = data.reshape((len(data), np.product(np.shape(data)[1:])))

    # compute umap graph
    umap_graph = build_fuzzy_simplicial_set(data, y=Y_masked, n_neighbors=n_neighbors,)"""

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
    dataset="cifar10",
    intersection=False,
    n_neighbors=15,
    use_last_layer=True,
):

    if learned_metric:

        # model to grab last layer activations
        last_layer_class = tf.keras.models.Model(
            classifier.input,
            [classifier.get_layer(name=classifier.layers[-2].name).get_output_at(0)],
        )

        # get encoder activations for X_train
        enc_z = encoder.predict(X_train)

        if use_last_layer:
            # get last layer activations for X_train
            last_layer_z = last_layer_class.predict(enc_z)
        else:
            last_layer_z = enc_z

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

        data = last_layer_z[high_confidence_mask]
        # flatten if needed
        if len(np.shape(data)) > 2:
            data = data.reshape((len(data), np.product(np.shape(data)[1:])))

        # compute umap graph
        umap_graph = build_fuzzy_simplicial_set(
            data, y=Y_masked, n_neighbors=n_neighbors,
        )
        if intersection:
            data_x = flatten(X_train)
            # flatten if needed
            if len(np.shape(data_x)) > 2:
                data_x = data_x.reshape((len(data_x), np.product(np.shape(data_x)[1:])))

            # compute umap graph of x
            umap_graph_x = build_fuzzy_simplicial_set(
                data_x, y=Y_masked, n_neighbors=n_neighbors,
            )

            # compute intersection
            # umap_graph = umap_graph + umap_graph_x / 2 #
            umap_graph = umap_graph + umap_graph_x - umap_graph * umap_graph_x

        edge_dataset = prepare_edge_dataset(
            umap_graph,
            augmented,
            batch_size,
            labeled_dataset,
            X_train_hc,
            Y_masked,
            dims,
            dataset=dataset,
        )

    else:

        data = flatten(X_train)
        # flatten if needed
        if len(np.shape(data)) > 2:
            data = data.reshape((len(data), np.product(np.shape(data)[1:])))

        # compute umap graph of x
        umap_graph_x = build_fuzzy_simplicial_set(
            data, y=Y_masked, n_neighbors=n_neighbors,
        )

        edge_dataset = prepare_edge_dataset(
            umap_graph_x,
            augmented,
            batch_size,
            labeled_dataset,
            X_train,
            Y_masked,
            dims,
            dataset=dataset,
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


# Taken from https://github.com/tanzhenyu/image_augmentation/blob/master/image_augmentation/image/image_ops.py
IMAGE_DTYPES = [tf.uint8, tf.float32, tf.float16, tf.float64]


@tf.function
def blend(image1, image2, factor, name=None):
    """Blends an image with another using `factor`.
    Args:
        image1: An int or float tensor of shape `[height, width, num_channels]`.
        image2: An int or float tensor of shape `[height, width, num_channels]`.
        factor: A 0-D float tensor or single floating point value depicting
            a weight above 0.0 for combining the example_images.
        name: An optional string for name of the operation.
    Returns:
        A tensor with same shape and type as that of `image1` and `image2`.
    """
    _check_image_dtype(image1)
    _check_image_dtype(image2)
    assert (
        image1.dtype == image2.dtype
    ), "image1 type should exactly match type of image2"

    if factor == 0.0:
        return image1
    elif factor == 1.0:
        return image2
    else:
        with tf.name_scope(name or "blend"):
            orig_dtype = image2.dtype

            image1, image2 = (
                tf.image.convert_image_dtype(image1, tf.float32),
                tf.image.convert_image_dtype(image2, tf.float32),
            )
            scaled_diff = (image2 - image1) * factor

            blended_image = image1 + scaled_diff

            blended_image = tf.image.convert_image_dtype(
                blended_image, orig_dtype, saturate=True
            )
            return blended_image


@tf.function
def brightness(image, magnitude, name=None):
    """Adjusts the `magnitude` of brightness of an `image`.
    Args:
        image: An int or float tensor of shape `[height, width, num_channels]`.
        magnitude: A 0-D float tensor or single floating point value above 0.0.
        name: An optional string for name of the operation.
    Returns:
        A tensor with same shape and type as that of `image`.
    """
    _check_image_dtype(image)

    with tf.name_scope(name or "brightness"):
        dark = tf.zeros_like(image)
        bright_image = blend(dark, image, magnitude)
        return bright_image


@tf.function
def cutout(image, size=16, color=None, name=None):
    """This is an implementation of Cutout as described in "Improved
    Regularization of Convolutional Neural Networks with Cutout" by
    DeVries & Taylor (https://arxiv.org/abs/1708.04552).
    It applies a random square patch of specified `size` over an `image`
    and by replacing those pixels with value of `color`.
    Args:
        image: An int or float tensor of shape `[height, width, num_channels]`.
        size: A 0-D int tensor or single int value that is divisible by 2.
        color: A single pixel value (grayscale) or tuple of 3 values (RGB),
            in case a single value is used for RGB image the value is tiled.
            Gray color (128) is used by default.
        name: An optional string for name of the operation.
    Returns:
        A tensor with same shape and type as that of `image`.
    """
    _check_image_dtype(image)

    with tf.name_scope(name or "cutout"):
        image_shape = tf.shape(image)
        height, width, channels = image_shape[0], image_shape[1], image_shape[2]

        loc_x = tf.random.uniform((), 0, width, tf.int32)
        loc_y = tf.random.uniform((), 0, height, tf.int32)

        ly, lx = tf.maximum(0, loc_y - size // 2), tf.maximum(0, loc_x - size // 2)
        uy, ux = (
            tf.minimum(height, loc_y + size // 2),
            tf.minimum(width, loc_x + size // 2),
        )

        gray = tf.constant(128)
        if color is None:
            if image.dtype == tf.uint8:
                color = tf.repeat(gray, channels)
            else:
                color = tf.repeat(tf.cast(gray, tf.float32) / 255.0, channels)
        else:
            color = tf.convert_to_tensor(color)
        color = tf.cast(color, image.dtype)

        cut = tf.ones((uy - ly, ux - lx, channels), image.dtype)

        top = image[0:ly, 0:width]
        between = tf.concat(
            [image[ly:uy, 0:lx], cut * color, image[ly:uy, ux:width]], axis=1
        )
        bottom = image[uy:height, 0:width]

        cutout_image = tf.concat([top, between, bottom], axis=0)
        return cutout_image


def _check_image_dtype(image):
    assert image.dtype in IMAGE_DTYPES, (
        "image with " + str(image.dtype) + " is not supported for this operation"
    )


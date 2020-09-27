from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
import numpy as np
from tfumap.paths import ensure_dir, MODEL_DIR, DATA_DIR
import gzip
import pickle
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
from sklearn.datasets import make_moons


def load_MNIST(flatten=True):
    # load dataset
    (train_images, Y_train), (test_images, Y_test) = mnist.load_data()
    X_train = (train_images / 255.0).astype("float32")
    X_test = (test_images / 255.0).astype("float32")
    if flatten:
        X_train = X_train.reshape((len(X_train), np.product(np.shape(X_train)[1:])))
        X_test = X_test.reshape((len(X_test), np.product(np.shape(X_test)[1:])))
    else:
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
    # subset a validation set
    n_valid = 10000
    X_valid = X_train[-n_valid:]
    Y_valid = Y_train[-n_valid:]
    X_train = X_train[:-n_valid]
    Y_train = Y_train[:-n_valid]

    return X_train, X_test, X_valid, Y_train, Y_test, Y_valid


def load_FMNIST(flatten=True):

    # load dataset
    (train_images, Y_train), (test_images, Y_test) = fashion_mnist.load_data()
    X_train = (train_images / 255.0).astype("float32")
    X_test = (test_images / 255.0).astype("float32")
    if flatten:
        X_train = X_train.reshape((len(X_train), np.product(np.shape(X_train)[1:])))
        X_test = X_test.reshape((len(X_test), np.product(np.shape(X_test)[1:])))
    else:
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)

    # subset a validation set
    n_valid = 10000
    X_valid = X_train[-n_valid:]
    Y_valid = Y_train[-n_valid:]
    X_train = X_train[:-n_valid]
    Y_train = Y_train[:-n_valid]

    return X_train, X_test, X_valid, Y_train, Y_test, Y_valid


def load_CIFAR10(flatten=True):
    # load dataset
    (train_images, Y_train), (test_images, Y_test) = cifar10.load_data()
    X_train = (train_images / 255.0).astype("float32")
    X_test = (test_images / 255.0).astype("float32")
    if flatten:
        X_train = X_train.reshape((len(X_train), np.product(np.shape(X_train)[1:])))
        X_test = X_test.reshape((len(X_test), np.product(np.shape(X_test)[1:])))

    # subset a validation set
    n_valid = 10000
    X_valid = X_train[-n_valid:]
    Y_valid = Y_train[-n_valid:].flatten()
    X_train = X_train[:-n_valid]
    Y_train = Y_train[:-n_valid].flatten()
    Y_test = Y_test.flatten()
    return X_train, X_test, X_valid, Y_train, Y_test, Y_valid


def load_MACOSKO():
    """
    #dataset_address = 'http://file.biolab.si/opentsne/macosko_2015.pkl.gz'
    # https://opentsne.readthedocs.io/en/latest/examples/01_simple_usage/01_simple_usage.html
    # also see https://github.com/berenslab/rna-seq-tsne/blob/master/umi-datasets.ipynb

    Returns
    -------
    [type]
        [description]
    """
    with gzip.open(DATA_DIR / "macosko_2015.pkl.gz", "rb") as f:
        data = pickle.load(f)

    x = data["pca_50"]
    y = data["CellType1"].astype(str)

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    np.shape(X_train)

    n_valid = 10000
    X_valid = X_train[-n_valid:]
    Y_valid = Y_train[-n_valid:]
    X_train = X_train[:-n_valid]
    Y_train = Y_train[:-n_valid]

    enc = OrdinalEncoder()
    Y_train = enc.fit_transform([[i] for i in Y_train]).flatten()
    Y_valid = enc.fit_transform([[i] for i in Y_valid]).flatten()
    Y_test = enc.fit_transform([[i] for i in Y_test]).flatten()

    return X_train, X_test, X_valid, Y_train, Y_test, Y_valid


def load_CASSINS():

    syllable_df = pd.read_pickle(DATA_DIR / "cassins" / "cassins.pickle")

    top_labels = (
        pd.DataFrame(
            {
                i: [np.sum(syllable_df.labels.values == i)]
                for i in syllable_df.labels.unique()
            }
        )
        .T.sort_values(by=0, ascending=False)[:20]
        .T
    )

    sylllable_df = syllable_df[syllable_df.labels.isin(top_labels.columns)]
    sylllable_df = sylllable_df.reset_index()

    sylllable_df["subset"] = "train"
    sylllable_df.loc[:1000, "subset"] = "valid"
    sylllable_df.loc[1000:1999, "subset"] = "test"

    Y_train = np.array(list(sylllable_df.labels.values[sylllable_df.subset == "train"]))
    Y_valid = np.array(list(sylllable_df.labels.values[sylllable_df.subset == "valid"]))
    Y_test = np.array(list(sylllable_df.labels.values[sylllable_df.subset == "test"]))

    X_train = np.array(
        list(sylllable_df.spectrogram.values[sylllable_df.subset == "train"])
    )  # / 255.
    X_valid = np.array(
        list(sylllable_df.spectrogram.values[sylllable_df.subset == "valid"])
    )  # / 255.
    X_test = np.array(
        list(sylllable_df.spectrogram.values[sylllable_df.subset == "test"])
    )  # / 255.

    enc = OrdinalEncoder()
    Y_train = enc.fit_transform([[i] for i in Y_train]).flatten()
    Y_valid = enc.fit_transform([[i] for i in Y_valid]).flatten()
    Y_test = enc.fit_transform([[i] for i in Y_test]).flatten()

    return X_train, X_test, X_valid, Y_train, Y_test, Y_valid


def load_MOONS(n_train=2000, n_test=1000, n_valid=1000, noise=0.1, random_state=None):
    if random_state is not None:
        random_state_test = random_state + 1
        random_state_valid = random_state + 2
    else:
        random_state_test = random_state_valid = random_state
    X_train, Y_train = make_moons(
        n_samples=n_train, noise=noise, random_state=random_state
    )
    X_test, Y_test = make_moons(
        n_samples=n_test, noise=noise, random_state=random_state_test
    )
    X_valid, Y_valid = make_moons(
        n_samples=n_valid, noise=noise, random_state=random_state_valid
    )
    return X_train, X_test, X_valid, Y_train, Y_test, Y_valid


def mask_labels(X_train, Y_train, labels_per_class=10):
    """ grabs a subset dataset for labels_per_class*n_class elements
    """
    label_mask = np.concatenate(
        [np.where(Y_train == i)[0][:labels_per_class] for i in np.unique(Y_train)]
    )
    X_labeled = X_train[label_mask]
    Y_labeled = Y_train[label_mask]
    Y_masked = np.zeros(len(Y_train)) - 1
    Y_masked[label_mask] = Y_labeled
    return X_labeled, Y_labeled, Y_masked

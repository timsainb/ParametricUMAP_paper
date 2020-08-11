import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tfumap.umap import retrieve_tensors
from sklearn.decomposition import PCA
from tqdm.autonotebook import tqdm


def batch(x, batch_size=100):
    n_batch = int(np.ceil((len(x) / batch_size)))
    return [x[batch_size * i : batch_size * (i + 1)] for i in range(n_batch)]


def plot_umap_classif_results(
    model,
    X_valid,
    Y_valid,
    X_train,
    X_labeled,
    Y_labeled,
    batch_size,
    cmap="coolwarm",
    cmap2="bwr",
):
    # get loss dataframe from tensorboard
    try:
        loss_df = retrieve_tensors(model.tensorboard_logdir)
        losses_exist = True
    except ValueError as e:
        print(e)
        losses_exist = False

    # embed data
    embedding_valid = embed_data(X_valid, model, batch_size)
    embedding_train = embed_data(X_train, model, batch_size)

    # encode data
    batched_X_valid = batch(X_valid)
    latent_valid = np.vstack(
        [model.encoder(i) for i in tqdm(batched_X_valid, leave=False)]
    )
    # latent_valid = model.encoder(X_valid)
    pca = PCA()
    z_valid = pca.fit_transform(latent_valid)

    # latent_lab = model.encoder(X_labeled)
    batched_X_lab = batch(X_labeled)
    latent_lab = np.vstack([model.encoder(i) for i in tqdm(batched_X_lab, leave=False)])
    z_lab = pca.transform(latent_lab)

    # plot
    fig, axs = plt.subplots(ncols=5, figsize=(26, 4))
    ax = axs[0]
    # ax.scatter(
    #    embedding_train[:, 0],
    #    embedding_train[:, 1],
    #    c="grey",
    #    cmap=cmap,  # "tab10"
    #    s=2,
    #    alpha=0.25,
    #    rasterized=True,
    # )

    ax.scatter(
        embedding_valid[:, 0],
        embedding_valid[:, 1],
        c=Y_valid,
        cmap=cmap,  # "tab10"
        s=2,
        alpha=0.25,
        rasterized=True,
    )
    ax.set_title("Projection data", fontsize=24)
    ax.axis("equal")
    ax = axs[1]
    ax.scatter(
        z_valid[:, 0],
        z_valid[:, 1],
        c=Y_valid.astype(int)[: len(embedding_valid)],
        cmap=cmap,  # "tab10"
        s=2,
        alpha=0.25,
        rasterized=True,
    )
    ax.scatter(
        z_lab[:, 0],
        z_lab[:, 1],
        c=Y_labeled,
        cmap=cmap2,  # "tab10"
        s=100,
        alpha=1,
        rasterized=True,
    )

    ax.set_title("PCA of enc.", fontsize=24)
    ax.axis("equal")

    if losses_exist:
        ax = axs[2]
        sns.lineplot(
            x="step",
            y="val",
            hue="group",
            data=loss_df[loss_df.variable == "umap_loss"],
            ci=None,
            ax=ax,
        )
        ax.legend(loc="upper right")
        ax.set_xscale("log")
        ax.set_title("UMAP loss", fontsize=24)
        ax.set_ylabel("Cross Entropy")
        ax = axs[3]
        sns.lineplot(
            x="step",
            y="val",
            hue="group",
            data=loss_df[loss_df.variable == "classif_loss"],
            ci=None,
            ax=ax,
        )
        ax.legend(loc="upper right")
        # ax.set_yscale("log")
        ax.set_title("Classif loss", fontsize=24)
        ax.set_ylabel("Cross Entropy")
        ax = axs[4]
        sns.lineplot(
            x="step",
            y="val",
            hue="group",
            data=loss_df[loss_df.variable == "classif_acc"],
            ci=None,
            ax=ax,
        )
        ax.legend(loc="lower right")
        ax.set_title("Classif acc", fontsize=24)
        ax.set_ylabel("Acc")
    plt.show()


def plot_results(
    model, X_valid, Y_valid, X_train, X_labeled, Y_labeled, batch_size, dims
):

    # get embedding
    nbatches = np.floor(len(X_valid) / batch_size).astype("int")
    valid_x = (
        X_valid[: nbatches * batch_size].reshape([nbatches, batch_size] + list([dims]))
    ).astype("float32")

    embedding_train = np.vstack(
        model.project_epoch(tf.data.Dataset.from_tensor_slices((valid_x)))
    )
    embedding = embedding_train
    projection_type = "training_data"
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(20, 10))

    # plot embedding
    ax = axs.flatten()[0]
    ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=Y_valid.astype(int)[: len(embedding)],
        cmap="coolwarm",  # "tab10"
        s=2,
        alpha=0.25,
        rasterized=True,
    )
    ax.set_title(projection_type, fontsize=24)
    ax.axis("equal")

    # plot loss
    for i, element in zip(
        range(1, 5),
        ["attraction_loss", "repellant_loss", "umap_loss", "classifier_loss"],
    ):
        ax = axs.flatten()[i]

        sns.lineplot(
            x="epoch", y=element, hue="type_", data=model.loss_df, ax=ax, ci=None
        )
        ax.set_title(element, fontsize=24)
        ax.set_yscale("log")
    ax = get_decision_contour(
        axs.flatten()[5], model, X_valid, X_train, X_labeled, Y_labeled
    )
    plt.show()


def embed_data(X, model, batch_size):
    """ embed a set of points in X to Z
    """
    n_batch = int(np.ceil(len(X) / batch_size))
    return np.vstack(
        [
            model.embedder(
                model.encoder(np.array(X[(i) * batch_size : (i + 1) * batch_size, :]))
            )
            for i in range(n_batch)
        ]
    )


def make_contour(model, samplerate=100):
    x_span = np.linspace(-2, 3, samplerate)
    y_span = np.linspace(-1, 1.5, samplerate)
    xx, yy = np.meshgrid(x_span, y_span)
    X_grid = np.array([xx.ravel(), yy.ravel()]).T
    decision_grid = classify_data(X_grid, model, batch_size=100)
    return xx, yy, decision_grid


def classify_data(X, model, batch_size):
    """ Classify a set of points X
    """
    n_batch = int(np.ceil(len(X) / batch_size))
    predictions = np.vstack(
        [
            model.classifier(
                model.encoder(np.array(X[(i) * batch_size : (i + 1) * batch_size, :]))
            )
            for i in range(n_batch)
        ]
    )
    predictions = tf.nn.softmax(predictions).numpy()
    return predictions[:, 1] - predictions[:, 0]


def get_decision_contour(
    ax,
    model,
    X_valid,
    X_train,
    X_labeled,
    Y_labeled,
    samplerate=100,
    color="grey",
    s=1,
    decision_alpha=0.5,
):
    z = embed_data(X_valid, model, 100)
    xx, yy, decision_grid = make_contour(model, samplerate=samplerate)

    ax.contourf(
        xx,
        yy,
        decision_grid.reshape((samplerate, samplerate)),
        cmap=plt.cm.coolwarm,
        alpha=decision_alpha,
        levels=np.linspace(-1, 1, 10),
    )

    ax.scatter(X_train[:, 0], X_train[:, 1], color=color, alpha=1, s=s)
    ax.scatter(X_labeled[:, 0], X_labeled[:, 1], c=Y_labeled, cmap=plt.cm.bwr, s=100)
    ax.set_title("Decision contour")
    return ax

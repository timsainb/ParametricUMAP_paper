### based on https://github.com/kylemcdonald/Parametric-t-SNE/blob/master/Parametric%20t-SNE%20(Keras).ipynb


import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
from tqdm.autonotebook import tqdm
import tensorflow as tf


def Hbeta(D, beta):
    """Computes the Gaussian kernel values given a vector of
    squared Euclidean distances, and the precision of the Gaussian kernel.
    The function also computes the perplexity (P) of the distribution."""
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta * np.sum(np.multiply(D, P)) / sumP
    P = P / sumP
    return H, P


def x2p(X, u=15, tol=1e-4, print_iter=500, max_tries=50, verbose=0):
    """
    % X2P Identifies appropriate sigma's to get kk NNs up to some tolerance 
    %
    %   [P, beta] = x2p(xx, kk, tol)
    % 
    % Identifies the required precision (= 1 / variance^2) to obtain a Gaussian
    % kernel with a certain uncertainty for every datapoint. The desired
    % uncertainty can be specified through the perplexity u (default = 15). The
    % desired perplexity is obtained up to some tolerance that can be specified
    % by tol (default = 1e-4).
    % The function returns the final Gaussian kernel in P, as well as the 
    % employed precisions per instance in beta.
    %
    """

    # Initialize some variables
    n = X.shape[0]  # number of instances
    P = np.zeros((n, n))  # empty probability matrix
    beta = np.ones(n)  # empty precision vector
    logU = np.log(u)  # log of perplexity (= entropy)

    # Compute pairwise distances
    if verbose > 0:
        print("Computing pairwise distances...")
    sum_X = np.sum(np.square(X), axis=1)
    # note: translating sum_X' from matlab to numpy means using reshape to add a dimension
    D = sum_X + sum_X[:, None] + -2 * X.dot(X.T)

    # Run over all datapoints
    if verbose > 0:
        print("Computing P-values...")
    for i in range(n):
        if verbose > 1 and print_iter and i % print_iter == 0:
            print("Computed P-values {} of {} datapoints...".format(i, n))

        # Set minimum and maximum values for precision
        betamin = float("-inf")
        betamax = float("+inf")

        # Compute the Gaussian kernel and entropy for the current precision
        indices = np.concatenate((np.arange(0, i), np.arange(i + 1, n)))
        Di = D[i, indices]
        H, thisP = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while abs(Hdiff) > tol and tries < max_tries:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i]
                if np.isinf(betamax):
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if np.isinf(betamin):
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2

            # Recompute the values
            H, thisP = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, indices] = thisP

    if verbose > 0:
        print("Mean value of sigma: {}".format(np.mean(np.sqrt(1 / beta))))
        print("Minimum value of sigma: {}".format(np.min(np.sqrt(1 / beta))))
        print("Maximum value of sigma: {}".format(np.max(np.sqrt(1 / beta))))

    return P, beta


def compute_joint_probabilities(
    samples, batch_size=5000, d=2, perplexity=30, tol=1e-5, verbose=0
):
    """ This function computes the probababilities in X, split up into batches
    % Gaussians employed in the high-dimensional space have the specified
    % perplexity (default = 30). The number of degrees of freedom of the
    % Student-t distribution may be specified through v (default = d - 1).
    """
    v = d - 1

    # Initialize some variables
    n = samples.shape[0]
    batch_size = min(batch_size, n)

    # Precompute joint probabilities for all batches
    if verbose > 0:
        print("Precomputing P-values...")
    batch_count = int(n / batch_size)
    P = np.zeros((batch_count, batch_size, batch_size))
    # for each batch of data
    for i, start in enumerate(tqdm(range(0, n - batch_size + 1, batch_size))):
        # select batch
        curX = samples[start : start + batch_size]
        # compute affinities using fixed perplexity
        P[i], _ = x2p(curX, perplexity, tol, verbose=verbose)
        # make sure we don't have NaN's
        P[i][np.isnan(P[i])] = 0
        # make symmetric
        P[i] = P[i] + P[i].T  # / 2
        # obtain estimation of joint probabilities
        P[i] = P[i] / P[i].sum()
        P[i] = np.maximum(P[i], np.finfo(P[i].dtype).eps)

    return P


def z2p(z, d, n, eps=10e-15):
    """ Computes the low dimensional probability
    """
    v = d - 1
    sum_act = tf.math.reduce_sum(tf.math.square(z), axis=1)
    Q = K.reshape(sum_act, [-1, 1]) + -2 * tf.keras.backend.dot(z, tf.transpose(z))
    Q = (sum_act + Q) / v
    Q = tf.math.pow(1 + Q, -(v + 1) / 2)
    Q *= 1 - np.eye(n)
    Q /= tf.math.reduce_sum(Q)
    Q = tf.math.maximum(Q, eps)
    return Q


def tsne_loss(d, batch_size, eps=10e-15):
    # v = d - 1.0
    def loss(P, Z):
        """ KL divergence
        P is the joint probabilities for this batch (Keras loss functions call this y_true)
        Z is the low-dimensional output (Keras loss functions call this y_pred)
        """
        Q = z2p(Z, d, n=batch_size, eps=eps)
        return tf.math.reduce_sum(P * tf.math.log((P + eps) / (Q + eps)))

    return loss

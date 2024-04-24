# Slightly modified example taken from http://krasserm.github.io/2018/03/19/gaussian-processes/

import numpy as np

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.

    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


### plot gp
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D


# ------------------------------------------
#  GPs for regression utils
# ------------------------------------------


def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[], i=None):

    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    plt.plot(X, mu, label='Mean')
    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label=f'Sample {i+1}')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'rx')
    plt.legend()

    plt.savefig(f'gp_{i}.png')

# Finite number of points
X = np.arange(-5, 5, 0.2).reshape(-1, 1)

# Mean and covariance of the prior
mu = np.zeros(X.shape)
cov = kernel(X, X)

# Draw three samples from the prior
samples = np.random.multivariate_normal(mu.ravel(), cov, 3)


from numpy.linalg import inv

def posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0):
    """
    Computes the suffifient statistics of the posterior distribution 
    from m training data X_train and Y_train and n new inputs X_s.

    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.

    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    """
    K = kernel(X_train, X_train, l, sigma_f)
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)

    # Equation (7)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (8)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s

# Noise free training data
X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
Y_train = np.sin(X_train)

# Compute mean and covariance of the posterior distribution
mu_s, cov_s = posterior(X, X_train, Y_train)

samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)

import matplotlib.pyplot as plt

kernel_params = [
    (1.5, 1.0),
    (0.5, 1.0),
    (1.0, 0.5),
    (1.0, 2.0),
]

plt.figure(figsize=(12, 5))

for i, (l, sigma_f) in enumerate(kernel_params):
    plt.close()
    mu_s, cov_s = posterior(X, X_train, Y_train, l=l, sigma_f=sigma_f, sigma_y=0)
    plt.title(f'l = {l}, sigma_f = {sigma_f}')
    plt.ylim(-5, 5)
    plot_gp(mu_s, cov_s, X, X_train=X_train, Y_train=Y_train, i=i)

import subprocess

# Combine gp_0.png and gp_1.png side by side
subprocess.call(["convert", "+append", "gp_0.png", "gp_1.png", "lengthscale.png"])

# Combine gp_2.png and gp_3.png side by side
subprocess.call(["convert", "+append", "gp_2.png", "gp_3.png", "sigma_f.png"])

# Stack buffer_1.png below buffer_0.png
subprocess.call(["convert", "-append", "lengthscale.png", "sigma_f.png", "kernel-parameters.png"])

"""A function that calculates obj & bhv for a batch of x_solutions (ie robot arm configurations)"""

import time
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange, tqdm
from numpy import float64

### Custom Scripts ###
from utils.pprint import pprint

from config import Config
config = Config(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))
SOL_DIMENSION = config.SOL_DIMENSION

def simulate_obj(samples, link_lengths = None):
    """Returns obj's for a batch of samples.

    Args:
        samples (np.ndarray): A (batch_size, dim) array where each row
            contains the joint angles for the arm. `dim` will always be 12
            in this tutorial.
        link_lengths (np.ndarray): A (dim,) array with the lengths of each
            arm link (this will always be an array of ones in the tutorial).
    Returns:
        objs (np.ndarray): (batch_size,) array of objectives.
        bhv (np.ndarray): (batch_size, 2) array of bhv.
    """

    if link_lengths == None: link_lengths=np.ones(SOL_DIMENSION)

    if not isinstance(samples, np.ndarray):
        samples = samples.numpy()
        pprint(samples)

    objs = -np.std(samples, axis=1)

    return objs


def simulate_bhv(samples, link_lengths = None):
    """Returns bhv's for a batch of samples.

    Args:
        samples (np.ndarray): A (batch_size, dim) array where each row
            contains the joint angles for the arm. `dim` will always be 12
            in this tutorial.
        link_lengths (np.ndarray): A (dim,) array with the lengths of each
            arm link (this will always be an array of ones in the tutorial).
    Returns:
        objs (np.ndarray): (batch_size,) array of objectives.
        bhv (np.ndarray): (batch_size, 2) array of bhv.
    """

    if link_lengths == None: link_lengths=np.ones(SOL_DIMENSION)

    # theta_1, theta_1 + theta_2, ...
    cum_theta = np.cumsum(samples, axis=1)

    # l_1 * cos(theta_1), l_2 * cos(theta_1 + theta_2), ...
    x_pos = link_lengths[None] * np.cos(cum_theta)
    # l_1 * sin(theta_1), l_2 * sin(theta_1 + theta_2), ...
    y_pos = link_lengths[None] * np.sin(cum_theta)

    bhv = np.concatenate(
        (
            np.sum(x_pos, axis=1, keepdims=True),
            np.sum(y_pos, axis=1, keepdims=True),
        ),
        axis=1,
    )

    return bhv


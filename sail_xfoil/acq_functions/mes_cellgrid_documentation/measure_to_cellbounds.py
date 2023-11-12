"""
This script demonstrates how sobol samples are drawn according to bin ranges


"""

import numpy as np
from ribs.emitters import GaussianEmitter
from ribs.archives import GridArchive

from chaospy import create_sobol_samples

from utils.pprint_nd import pprint

BATCH_SIZE = 10

BHV_DIM = 2
BHV_NUMBER_BINS = [5,5]
BHV_VALUE_RANGES = [(5,10),(-10,-5)]

SOLUTION_DIM = 4
SOL_VALUE_RANGE = [(1,2), (5,10),(-10,-5),(-20,-10)]


def measure_to_cellbounds(print_flag=True):
    """Converts a measure batch to a batch of cell bounds corresponding to a given archive"""

    ranges = np.array(SOL_VALUE_RANGE)

    def uniform_sample():
        uniform_sample = np.random.uniform(ranges[:, 0], ranges[:, 1], SOLUTION_DIM)
        return uniform_sample

    init_samples = np.array([uniform_sample() for i in range(BATCH_SIZE)])
    init_samples_bhv = init_samples[:,1:3]

    pprint(init_samples)
    print()

    obj_archive = GridArchive(
    solution_dim=SOLUTION_DIM,                      # Dimension of solution vector
    dims=BHV_NUMBER_BINS,                # Discretization of behavioral bins
    ranges=BHV_VALUE_RANGES,             # Possible values for behavior vector
    qd_score_offset=-600,
    threshold_min = -1,
    seed=1
    )

    obj_archive.add(init_samples, np.ones(BATCH_SIZE), init_samples_bhv)

    archive_indices = obj_archive.index_of(init_samples_bhv)
    idx = obj_archive.int_to_grid_index(archive_indices)

    # define a vector that contains 10 vectors, each of which contains 2 vectors of length 2
    # the first vector of length 2 contains the lower and upper bounds of the x dimension  
    # the second vector of length 2 contains the lower and upper bounds of the y dimension
    val_rngs = np.empty((BATCH_SIZE,BHV_DIM,2))
    
    for i in range(BATCH_SIZE):

        measure_0_idx, measure_1_idx = idx[i]
        
        lower_0_bound = obj_archive.boundaries[0][measure_0_idx]
        upper_0_bound = obj_archive.boundaries[0][measure_0_idx+1]

        lower_1_bound = obj_archive.boundaries[1][measure_1_idx]
        upper_1_bound = obj_archive.boundaries[1][measure_1_idx+1]

        val_range_0 = (lower_0_bound, upper_0_bound)
        val_range_1 = (lower_1_bound, upper_1_bound)

        val_rng = np.array([val_range_0, val_range_1])
        val_rngs[i] = val_rng
    
    if print_flag:
        print(val_rngs)
    return val_rngs


def cellbounds_to_value_range():
    """
    Converts a batch of cell bounds to a batch of value ranges.
    
    In this example, index 1/2 inside SOL_VALUE_RANGE 
    correspond to the x/y dimension of the behavior space
    """
    
    bhv_val_rngs = measure_to_cellbounds(print_flag=False)
    sol_val_rngs = np.empty((BATCH_SIZE,SOLUTION_DIM,2))
    
    for i in range(BATCH_SIZE):

        low_0, up_0 = bhv_val_rngs[i][0]
        low_1, up_1 = bhv_val_rngs[i][1]

        bhv_rngs_0 = (low_0, up_0)
        bhv_rngs_1 = (low_1, up_1)

        sol_val_rng_i = SOL_VALUE_RANGE.copy()
        sol_val_rng_i[1] = bhv_rngs_0
        sol_val_rng_i[2] = bhv_rngs_1

        sol_val_rngs[i] = np.array(sol_val_rng_i)

    print(sol_val_rngs)
    return sol_val_rngs


def cellbounds_to_sobol_sample():
    """Creates a Sobol Sample inside each cell of the queried samples"""

    sol_val_rngs = cellbounds_to_value_range()

    mes_cellgrids = np.empty((BATCH_SIZE,1000,SOLUTION_DIM))
    for i in range(sol_val_rngs.shape[0]):
        sobol_cellgrid = create_sobol_samples(order=1000, dim=SOLUTION_DIM, seed=123)
        sobol_cellgrid = sobol_cellgrid.T
        sol_val_rng = sol_val_rngs[i]

        lower_bounds = sol_val_rng[:, 0]
        upper_bounds = sol_val_rng[:, 1]

        mes_cellgrid_i = sobol_cellgrid * (upper_bounds - lower_bounds) + lower_bounds
        mes_cellgrids[i] = mes_cellgrid_i

    return mes_cellgrids


if __name__ == '__main__':
    cellbounds_to_sobol_sample()


import numpy as np
from ribs.emitters import GaussianEmitter
from ribs.archives import GridArchive

from chaospy import create_sobol_samples

from utils.pprint_nd import pprint

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')


BHV_DIM = 2
SOLUTION_DIM = 5
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE


def archive_cellbounds(archive: GridArchive):
    """Calculates cellbounds for all cells of a given archive"""

    # 625 bins in 25x25 grid archive
    n_bins = np.prod(archive.dims) 

    archive_indices = range(n_bins)
    idx = archive.int_to_grid_index(archive_indices)

    cell_bounds = np.empty((n_bins, BHV_DIM, 2))

    boundaries_0 = archive.boundaries[0]
    boundaries_1 = archive.boundaries[1]

    print("\nboundaries_0:\n", boundaries_0)
    print("\nboundaries_1:\n", boundaries_1)

    for i in archive_indices:

        measure_0_idx, measure_1_idx = idx[i]
        
        lower_0_bound = boundaries_0[measure_0_idx]
        upper_0_bound = boundaries_0[measure_0_idx+1]

        lower_1_bound = boundaries_1[measure_1_idx]
        upper_1_bound = boundaries_1[measure_1_idx+1]

        cell_bounds_0 = (lower_0_bound, upper_0_bound)
        cell_bounds_1 = (lower_1_bound, upper_1_bound)

        cell_bounds_i = np.array([cell_bounds_0, cell_bounds_1])
        cell_bounds[i] = cell_bounds_i

    print("\ncell_bounds:\n", cell_bounds)
    
    return cell_bounds


def cellbounds_to_valuerange(cell_bounds):
    """
    Converts a batch of cell bounds to a batch of solution value ranges.
    
    In this example, index 2/3 inside SOL_VALUE_RANGE 
    correspond to the x/y dimension of the behavior space
    """

    n_cells = cell_bounds.shape[0]
    
    cell_value_rngs = np.empty((n_cells, SOLUTION_DIM, 2))
    
    for i in range(n_cells):

        low_0, up_0 = cell_bounds[i][0]
        low_1, up_1 = cell_bounds[i][1]

        bhv_rngs_0 = (low_0, up_0)
        bhv_rngs_1 = (low_1, up_1)

        sol_val_rng_i = SOL_VALUE_RANGE.copy()
        sol_val_rng_i[2] = bhv_rngs_0
        sol_val_rng_i[3] = bhv_rngs_1

        cell_value_rngs[i] = np.array(sol_val_rng_i)

    print(f"\nValue Ranges:\n{cell_value_rngs}")
    return cell_value_rngs


def valueranges_to_cellgrid(valueranges):
    """Creates a Sobol Cellgrid for each cell of the requested archive"""

    n_cells = valueranges.shape[0]

    mes_cellgrids = np.empty((n_cells, 10000, SOLUTION_DIM))

    for i in range(n_cells):
        
        sobol_cellgrid = create_sobol_samples(order=10000, dim=SOLUTION_DIM, seed=123)
        sobol_cellgrid = sobol_cellgrid.T
        sol_val_rng = valueranges[i]

        print(f"\nShape: {sobol_cellgrid.shape}  Bin: {i}")
        print("Sobol Cellgrid:\n", sobol_cellgrid)

        lower_bounds = sol_val_rng[:, 0]
        upper_bounds = sol_val_rng[:, 1]

        mes_cellgrid_i = sobol_cellgrid * (upper_bounds - lower_bounds) + lower_bounds
        mes_cellgrids[i] = mes_cellgrid_i


        print(f"\nShape: {mes_cellgrid_i.shape}  Bin: {i}")
        print("MES Cellgrid:\n", mes_cellgrid_i)

        if i==0:
            lala=5
        if i==624:
            lala=5

    print(mes_cellgrids)
    return mes_cellgrids


if __name__ == "__main__":

    obj_archive = GridArchive(
        solution_dim=5,
        dims=[25,25],
        ranges=[(25,50),(-50,-25)],
        qd_score_offset=-600,
        threshold_min = -1337
    )

    cell_bounds = archive_cellbounds(obj_archive)

    SOL_VALUE_RANGE = [(1,2) , (3,4), (25,50), (-50,-25), (5,6)]
    valueranges = cellbounds_to_valuerange(cell_bounds)

    mes_cellgrids = valueranges_to_cellgrid(valueranges)


### Packages ###
from torch import float64, cuda, device, tensor
from botorch.acquisition import qMaxValueEntropy
from chaospy import create_sobol_samples
import numpy as np

### Custom Scripts ###w
from utils.pprint_nd import pprint

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
SIGMA_UCB = config.SIGMA_UCB
SOL_DIMENSION = config.SOL_DIMENSION
BHV_DIMENSION = config.BHV_DIMENSION
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE


def acq_mes(self, genomes):

    dev = device("cuda" if cuda.is_available() else "cpu")

    genomes = tensor(genomes, dtype=float64, device=dev)     # Shape: PARALLEL_BATCH_SIZE x SOL_DIMENSION
    transformed_genomes = genomes.unsqueeze(1)               # Shape: PARALLEL_BATCH_SIZE x 1 x SOL_DIMENSION

    acq_entropy_tensor = tensor(np.zeros((len(genomes), 1)), dtype=float64, device=dev)   # Shape: PARALLEL_BATCH_SIZE x 1
    for i in range(len(genomes)):

        cellgrid = assamble_cellgrid(self, genomes[i])

        cellgrid = tensor(cellgrid, dtype=float64, device=dev)   # Shape: 10000 x SOL_DIMENSION
        MES = qMaxValueEntropy(self.gp_model, cellgrid)

        acq_entropy = MES(transformed_genomes[i])
        acq_entropy_tensor[i] = acq_entropy

    mes_ndarray = acq_entropy_tensor.detach().numpy()

    return np.hstack(mes_ndarray)


def assamble_cellgrid(self, genome):
    """Creates a Sobol Cellgrid for a requested genome"""

    mes_cellgrid = np.empty((1, 10000, SOL_DIMENSION))

    genome_behavior = genome[1:3]
    genome_cell_index = self.obj_archive.index_of_single(genome_behavior)

    mes_cellgrid = self.mes_sobol_cellgrid.copy()
    bhv_cellgrid_i = self.bhv_sobol_cellgrids[genome_cell_index].copy()

    mes_cellgrid[:,1:3] = bhv_cellgrid_i

    return mes_cellgrid


def mes_sobol_cellgrids(self):

    """
    Creates a Sobol Cellgrid that can be used for all cells

        This function is called once before every MAP-Elites-Loop.
    
        Among non-measure dimensions, all sobol samples are equal.
        Therefore we need to draw only one sobol sample for all grids.

        Seperation of bhv_cellgrids and mes_cellgrids allows us to
        scale the behavior space independently from the solution space,
        which accelerates calculation, while reducing also reducing
        memory consumption significantly.

        Mes/Bhv Cellgrids are stored within the SailRunner class.
        Mes Cellgrid is constant across all bins.
        Bhv Cellgrid can be accessed by index.

        Therefore, we can rapidly assamble the final cellgrid
        for each sample within the MAP-Loop

    Returns:

        bhv_cellgrids : 625 bins x 10000 samples x 2 dimensions
        mes_cellgrid  :   1      x 10000 samples x 11 dimensions

    # how does the naive approach work? : https://github.com/patrickab/thesis/blob/master/sail_xfoil/acq_functions/mes_cellgrid_documentation/MES%20Sobol%20Cellgrids.pdf
    # why would this approach be naive? : https://github.com/patrickab/thesis/blob/master/sail_xfoil/acq_functions/mes_cellgrid_documentation/MES%20Sobol%20Cellgrids.mp4

    """
    sobol_cellgrid = create_sobol_samples(order=10000, dim=SOL_DIMENSION, seed=self.current_seed).T

    archive = self.obj_archive
    n_cells = np.prod(archive.dims)

    archive_indices = range(n_cells)
    idx = archive.int_to_grid_index(archive_indices)

    lower_bounds = np.array(SOL_VALUE_RANGE)[:, 0]
    upper_bounds = np.array(SOL_VALUE_RANGE)[:, 1]

    bhv_cellgrid = sobol_cellgrid[:, 1:3]
    mes_cellgrid = sobol_cellgrid * (upper_bounds - lower_bounds) + lower_bounds

    boundaries_0 = archive.boundaries[0]
    boundaries_1 = archive.boundaries[1]

    # 625 bins, 10000 samples, 2 dimensions
    bhv_cellgrids = np.empty((n_cells, 10000, BHV_DIMENSION))

    for i in range(n_cells):

        measure_0_idx, measure_1_idx = idx[i]

        cell_bounds_0 = (boundaries_0[measure_0_idx], boundaries_0[measure_0_idx+1])
        cell_bounds_1 = (boundaries_1[measure_1_idx], boundaries_1[measure_1_idx+1])

        cell_bounds_i = np.array([cell_bounds_0, cell_bounds_1])

        lower_bounds = cell_bounds_i[:, 0]
        upper_bounds = cell_bounds_i[:, 1]
        cell_bound_ranges = upper_bounds - lower_bounds

        bhv_cellgrid_i = bhv_cellgrid.copy()        
        bhv_cellgrid_i = bhv_cellgrid_i * cell_bound_ranges.T + lower_bounds   # scale sobol cellgrid to cellbounds
        bhv_cellgrids[i] = bhv_cellgrid_i                                      # insert bhv cellgrid into mes cellgrid

        verification = self.obj_archive.index_of(bhv_cellgrid_i)

        # verify if all samples are in the same cell
        if np.unique(verification).shape[0] != 1:
            raise ValueError("MES Sobol Cellgrid Error")
        
        # verify if all samples are in the correct cell
        if verification[0] != i:
            raise ValueError("MES Sobol Cellgrid Error")

    return bhv_cellgrids, mes_cellgrid
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

    cell_bounds = genome_cellbounds(self, genomes)
    cellgrids   = cellbounds_to_cellgrid(self, cell_bounds)

    genomes = tensor(genomes, dtype=float64, device=dev)     # Shape: PARALLEL_BATCH_SIZE x SOL_DIMENSION
    transformed_genomes = genomes.unsqueeze(1)               # Shape: PARALLEL_BATCH_SIZE x 1 x SOL_DIMENSION

    acq_entropy_tensor = tensor(np.zeros((len(genomes), 1)), dtype=float64, device=dev)   # Shape: PARALLEL_BATCH_SIZE x 1
    for i in range(len(genomes)):

        cellgrid = tensor(cellgrids[i], dtype=float64, device=dev)   # Shape: 10000 x SOL_DIMENSION
        MES = qMaxValueEntropy(self.gp_model, cellgrid)
        acq_entropy = MES(transformed_genomes[i])
        acq_entropy_tensor[i] = acq_entropy

    mes_ndarray = acq_entropy_tensor.detach().numpy()

    return np.hstack(mes_ndarray)


def cellbounds_to_cellgrid(self, cellbounds):
    """Creates a Sobol Cellgrid for each requested cell"""

    n_cells = cellbounds.shape[0]

    mes_cellgrids = np.empty((n_cells, 10000, SOL_DIMENSION))

    for i in range(n_cells):

        mes_cellgrid_i = self.mes_sobol_cellgrid.copy()
        bhv_cellgrid_i = self.bhv_sobol_cellgrid.copy()        

        lower_bounds = cellbounds[i, 0]
        upper_bounds = cellbounds[i, 1]

        bhv_cellgrid_i = bhv_cellgrid_i * (upper_bounds - lower_bounds) + lower_bounds   # scale sobol cellgrid to cellbounds
        mes_cellgrid_i[:, 1:3] = bhv_cellgrid_i                                       # insert bhv cellgrid into mes cellgrid

        mes_cellgrids[i] = mes_cellgrid_i                                             # store mes cellgrid

    return mes_cellgrids


def genome_cellbounds(self, genomes):

    """
    Calculates cellbounds for all queried genomes
    
    Assumes that all archives have equally sized and scaled bins.
    Assumes column 2 and 3 of genomes to be behavior dimensions.
    """

    n_genomes = genomes.shape[0] 

    cell_indices = self.obj_archive.index_of(genomes[:,1:3])
    idx = self.obj_archive.int_to_grid_index(cell_indices)

    cell_bounds = np.empty((n_genomes, BHV_DIMENSION, 2))

    boundaries_0 = self.obj_archive.boundaries[0]
    boundaries_1 = self.obj_archive.boundaries[1]

    for i in range(n_genomes):

        measure_0_idx, measure_1_idx = idx[i]
        
        lower_0_bound = boundaries_0[measure_0_idx]
        upper_0_bound = boundaries_0[measure_0_idx+1]

        lower_1_bound = boundaries_1[measure_1_idx]
        upper_1_bound = boundaries_1[measure_1_idx+1]

        cell_bounds_0 = (lower_0_bound, upper_0_bound)
        cell_bounds_1 = (lower_1_bound, upper_1_bound)

        cell_bounds_i = np.array([cell_bounds_0, cell_bounds_1])
        cell_bounds[i] = cell_bounds_i
    
    return cell_bounds


def mes_sobol_cellgrid(self):

    """
    Creates a Sobol Cellgrid that can be used for all cells

        This function is only called once inside each MAP-Elites-Loop.
    
        Among non-measure dimensions, all sobol samples are equal.
        Therefore we need to draw only one sobol sample for all grids.

        Seperation of bhv_cellgrids and mes_cellgrids allows us to
        scale the behavior space independently from the solution space,
        which accelerates calculation significantly.

        ###link to github###
    """

    sobol_cellgrid = create_sobol_samples(order=10000, dim=SOL_DIMENSION, seed=self.current_seed).T    

    bhv_cellgrid = sobol_cellgrid[:, 1:3]

    lower_bounds = np.array(SOL_VALUE_RANGE)[:, 0]
    upper_bounds = np.array(SOL_VALUE_RANGE)[:, 1]

    mes_cellgrid = sobol_cellgrid * (upper_bounds - lower_bounds) + lower_bounds

    return bhv_cellgrid, mes_cellgrid


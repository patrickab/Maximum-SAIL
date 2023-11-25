### Packages ###
from torch import float64, cuda, device, tensor
from botorch.acquisition import qMaxValueEntropy
from chaospy import create_sobol_samples
import numpy as np
import gc

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

    # if genomes is empty, return empty array
    if len(genomes) == 0:
        return np.array([])

    rng = np.random.default_rng(self.current_seed)
    cell_indices = self.obj_archive.index_of(genomes[:,1:3])
    cellbounds = self.bhv_cellbounds[cell_indices]
    solutionbounds = np.array(SOL_VALUE_RANGE)
    cell_solutionbounds = np.repeat(solutionbounds[np.newaxis,:,:], len(genomes), axis=0)    # create 8 copies of solutionbounds
    cell_solutionbounds[:, 1:3] = cellbounds                                                 # insert niche-specific cellbounds

    # mutate each genome 500 times using gaussian noise scaled to cell_solutionbounds
    genomes = np.repeat(genomes, 500, axis=0).reshape(len(genomes), 500, SOL_DIMENSION)   
    for i in range(len(genomes)):
        scaled_noise = rng.normal(scale=np.abs(0.6*(cell_solutionbounds[i,:,1] - cell_solutionbounds[i,:,0])), size=(500, SOL_DIMENSION))
        genomes[i] = np.clip(genomes[i] + scaled_noise, cell_solutionbounds[i,:,0], cell_solutionbounds[i,:,1])


    genomes_tensor = tensor(genomes, dtype=float64)     # Shape: 8 x BATCH_SIZE x SOL_DIMENSION
    transformed_genomes = genomes_tensor.unsqueeze(1)   # Shape: 1 x BATCH_SIZE x 1 x SOL_DIMENSION

    # calculate MES for each mutant batch & select best mutant
    acq_solution_tensor = tensor(np.zeros((len(genomes), SOL_DIMENSION)), dtype=float64)    # Shape: PARALLEL_BATCH_SIZE x 1
    acq_entropy_tensor = tensor(np.zeros((len(genomes), 1)), dtype=float64)                 # Shape: PARALLEL_BATCH_SIZE x 1
    for i in range(genomes.shape[0]):

        cellgrid = assamble_cellgrid(self, genomes_tensor[i,0])
        cellgrid = tensor(cellgrid, dtype=float64)      # Shape: 10000 x SOL_DIMENSION
        MES = qMaxValueEntropy(model=self.gp_model, candidate_set=cellgrid, num_y_samples=256)
        acq_entropy = MES(transformed_genomes[i].permute(1, 0, 2))
        
        elite_index = acq_entropy.argmax()
        acq_entropy_tensor[i] = acq_entropy[elite_index]
        acq_solution_tensor[i] = genomes_tensor[i,elite_index]

        indices = np.where(acq_entropy > 0.15)[0]
        best_solutions = genomes_tensor[i, indices]
        if best_solutions.shape[0] != 0:
            best_entropies = acq_entropy[indices]
            result_cell_indices = self.acq_archive.index_of(best_solutions[:,1:3])
            n_unique_cells = np.unique(result_cell_indices).shape[0]
            if n_unique_cells > 1:
                print("\nMULTIPLE HIGHPERFORMING MUTANTS")
                print("Number Unique Cells: ", n_unique_cells, "\n")
                if n_unique_cells > 4:
                    raise ValueError("n_unique_cells > 4")
                best_behavior = best_solutions[:,1:3]
                self.acq_archive.add(best_solutions.detach().numpy(), best_entropies.detach().numpy(), best_behavior.detach().numpy())

    # Store MES Elites in SailRunner class to use them inside the MAP-Loop
    self.mes_elites = acq_solution_tensor.detach().numpy()
    mes_ndarray = acq_entropy_tensor.detach().numpy()

    return np.hstack(mes_ndarray)


def simple_mes(self, genomes):

    print("\n\n\n\n\nentering simple_mes\n\n\n\n\n\n\n")
    # if genomes is empty, return empty array
    if len(genomes) == 0:
        return np.array([])

    genomes_tensor = tensor(genomes, dtype=float64)     # Shape: 8 x BATCH_SIZE x SOL_DIMENSION
    transformed_genomes = genomes_tensor.unsqueeze(1)   # Shape: 1 x BATCH_SIZE x 1 x SOL_DIMENSION

    # calculate MES for each mutant batch & select best mutant
    acq_solution_tensor = tensor(np.zeros((len(genomes), SOL_DIMENSION)), dtype=float64)    # Shape: PARALLEL_BATCH_SIZE x 1
    acq_entropy_tensor = tensor(np.zeros((len(genomes), 1)), dtype=float64)                 # Shape: PARALLEL_BATCH_SIZE x 1
    for i in range(genomes.shape[0]):

        cellgrid = assamble_cellgrid(self, genomes_tensor[i,0])
        cellgrid = tensor(cellgrid, dtype=float64)      # Shape: 10000 x SOL_DIMENSION
        MES = qMaxValueEntropy(model=self.gp_model, candidate_set=cellgrid, num_y_samples=256)
        acq_entropy = MES(transformed_genomes[i].permute(1, 0, 2))
        
        elite_index = acq_entropy.argmax()
        acq_entropy_tensor[i] = acq_entropy[elite_index]
        acq_solution_tensor[i] = genomes_tensor[i,elite_index]

    # Store MES Elites in SailRunner class to use them inside the MAP-Loop
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

        bhv_cellbounds : 625 bins x 2  dimensions x 2 boundaries
        bhv_cellgrids  : 625 bins x 10000 samples x 2 dimensions
        mes_cellgrid   :   1      x 10000 samples x 11 dimensions

    # how does the naive approach work? : https://github.com/patrickab/thesis/blob/master/sail_xfoil/acq_functions/mes_cellgrid_documentation/MES%500Sobol%500Cellgrids.pdf
    # why would this approach be naive? : https://github.com/patrickab/thesis/blob/master/sail_xfoil/acq_functions/mes_cellgrid_documentation/MES%500Sobol%500Cellgrids.mp4

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
    bhv_cellbounds = np.empty((n_cells, BHV_DIMENSION, 2))

    for i in range(n_cells):

        measure_0_idx, measure_1_idx = idx[i]

        cell_bounds_0 = (boundaries_0[measure_0_idx], boundaries_0[measure_0_idx+1])
        cell_bounds_1 = (boundaries_1[measure_1_idx], boundaries_1[measure_1_idx+1])

        cell_bounds_i = np.array([cell_bounds_0, cell_bounds_1])
        bhv_cellbounds[i] = cell_bounds_i

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

    return bhv_cellbounds, bhv_cellgrids, mes_cellgrid
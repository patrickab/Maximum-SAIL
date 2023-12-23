### Packages ###
import torch
from torch import float64, tensor
from botorch.acquisition import qMaxValueEntropy, qLowerBoundMaxValueEntropy
from botorch.optim import optimize_acqf
import numpy as np
import gc
import os

### Configurable Variables ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
MES_MUTANTS = config.MES_MUTANTS
MES_GRID_SIZE = config.MES_GRID_SIZE
NUM_MV_SAMPLES = config.NUM_MV_SAMPLES
SOL_DIMENSION = config.SOL_DIMENSION
BHV_DIMENSION = config.BHV_DIMENSION
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SIGMA_MUTANTS = config.SIGMA_MUTANTS


def acq_mes(self, gp_model, genomes, niche_restricted_update=False):
    """This function is used only in MES-Map-Elites and may not be up to date"""

    rng = np.random.default_rng(self.current_seed)
    cell_indices = self.acq_archive.index_of(genomes[:,1:3])    # Shape: BATCH_SIZE x 1 

    if niche_restricted_update:
        cellbounds = self.bhv_cellbounds[cell_indices]
    else:
        cellbounds = self.bhv_cellbounds_mutants[cell_indices]

    solutionbounds = np.array(SOL_VALUE_RANGE)
    cell_solutionbounds = np.repeat(solutionbounds[np.newaxis,:,:], len(genomes), axis=0)
    cell_solutionbounds[:, 1:3] = cellbounds

    # Generate n=MES_MUTANTS using Gaussian Noise
    genomes = np.repeat(genomes, MES_MUTANTS, axis=0).reshape(len(genomes), MES_MUTANTS, SOL_DIMENSION)   
    for i in range(len(genomes)):
        scaled_noise = rng.normal(
                            scale=np.abs(SIGMA_MUTANTS *(cell_solutionbounds[i,:,1] - cell_solutionbounds[i,:,0])), 
                            size=(MES_MUTANTS, SOL_DIMENSION))

        genomes[i] = np.clip(genomes[i] + scaled_noise, cell_solutionbounds[i,:,0], cell_solutionbounds[i,:,1])

    genomes_tensor = tensor(genomes, dtype=float64)          # Shape: 8 x BATCH_SIZE x SOL_DIMENSION
    transformed_genomes = genomes_tensor.unsqueeze(1)        # Shape: 1 x BATCH_SIZE x 1 x SOL_DIMENSION

    # calculate MES for each mutant batch & select best mutant
    acq_solution_tensor = tensor(np.zeros((len(genomes), SOL_DIMENSION)), dtype=float64)    # Shape: PARALLEL_BATCH_SIZE x 1
    acq_entropy_tensor = tensor(np.zeros((len(genomes), 1)), dtype=float64)                 # Shape: PARALLEL_BATCH_SIZE x 1
    for i in range(genomes.shape[0]):

        cellgrid = assamble_cellgrid(self, genomes_tensor[i,0])
        cellgrid = tensor(cellgrid, dtype=float64)                   # Shape: 4000 x SOL_DIMENSION
        MES = qLowerBoundMaxValueEntropy(model=gp_model, candidate_set=cellgrid, num_mv_samples=NUM_MV_SAMPLES)
        acq_entropy = MES(transformed_genomes[i].permute(1, 0, 2))

        elite_index = acq_entropy.argmax()
        acq_entropy_tensor[i] = acq_entropy[elite_index]
        acq_solution_tensor[i] = genomes_tensor[i,elite_index]

        self.acq_archive.add(genomes_tensor[i].detach().numpy(), acq_entropy.detach().numpy(), genomes_tensor[i][:,1:3].detach().numpy())

    # Store MES Elites in SailRunner class to use them inside the MAP-Loop
    self.mes_elites = acq_solution_tensor.detach().numpy()
    mes_ndarray = acq_entropy_tensor.detach().numpy()

    del cell_indices, cellbounds, solutionbounds, cell_solutionbounds, genomes, scaled_noise, genomes_tensor, transformed_genomes, acq_solution_tensor, acq_entropy_tensor, cellgrid, MES, acq_entropy, elite_index, rng
    gc.collect()

    return np.hstack(mes_ndarray)


def simple_mes(self, genomes):

    # if genomes is empty, return empty array
    if len(genomes) == 0:
        return np.array([])

    genomes_tensor = tensor(genomes, dtype=float64)     # Shape: 8 x BATCH_SIZE x SOL_DIMENSION
    transformed_genomes = genomes_tensor.unsqueeze(1)   # Shape: 1 x BATCH_SIZE x 1 x SOL_DIMENSION

    acq_solution_tensor = tensor(np.zeros((len(genomes), SOL_DIMENSION)), dtype=float64)    # Shape: PARALLEL_BATCH_SIZE x 1
    acq_entropy_tensor = tensor(np.zeros((len(genomes), 1)), dtype=float64)                 # Shape: PARALLEL_BATCH_SIZE x 1
    for i in range(genomes.shape[0]):

        cellgrid = assamble_cellgrid(self, genomes_tensor[i,0])
        cellgrid = tensor(cellgrid, dtype=float64)      # Shape: 4000 x SOL_DIMENSION
        MES = qLowerBoundMaxValueEntropy(model=self.gp_model, candidate_set=cellgrid, num_y_samples=100)
        acq_entropy = MES(transformed_genomes[i].permute(1, 0, 2))
        
        elite_index = acq_entropy.argmax()
        acq_entropy_tensor[i] = acq_entropy[elite_index]
        acq_solution_tensor[i] = genomes_tensor[i,elite_index]

    # Store MES Elites in SailRunner class to use them inside the MAP-Loop
    mes_ndarray = acq_entropy_tensor.detach().numpy()

    return np.hstack(mes_ndarray)


def assamble_cellgrid(self, genome):
    """Creates a Sobol Cellgrid for a requested genome"""

    mes_cellgrid = np.empty((1, MES_GRID_SIZE, SOL_DIMENSION))

    genome_behavior = genome[1:3]
    genome_cell_index = self.acq_archive.index_of_single(genome_behavior)

    mes_cellgrid = self.mes_sobol_cellgrid.copy()
    bhv_cellgrid_i = self.bhv_sobol_cellgrids[genome_cell_index].copy()

    mes_cellgrid[:,1:3] = bhv_cellgrid_i

    return mes_cellgrid


def optimize_mes(self, init_flag=False, map_flag=False):

    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    gp_model = self.gp_model
    n_samples = 10

    acq_elite_df = self.acq_archive.as_pandas(include_solutions=True)
    acq_elite_df = acq_elite_df.sample(frac=1, random_state=self.current_seed)
    acq_elite_df = acq_elite_df.head(n=n_samples)

    print("\nOptimize MES: n_samples", n_samples)

    sum_perc_improvement = 0

    genomes = acq_elite_df.solution_batch()
    objectives = acq_elite_df.objective_batch()

    if init_flag:
        self.acq_archive.clear()

    sum_perc_improvement = 0

    genomes = acq_elite_df.solution_batch()
    objectives = acq_elite_df.objective_batch()

    cell_indices = self.acq_archive.index_of(genomes[:,1:3])

    cellbounds = self.bhv_cellbounds[cell_indices]
    cellbounds[:,:1,0] = cellbounds[:,:1,0]
    cellbounds[:,:1,1] = cellbounds[:,:1,1]
    solutionbounds = np.array(SOL_VALUE_RANGE)
    cell_solutionbounds = np.repeat(solutionbounds[np.newaxis,:,:], len(genomes), axis=0)    # create copies of solutionbounds
    cell_solutionbounds[:, 1:3] = cellbounds                                                 # insert niche-specific cellbounds

    genomes_tensor = tensor(genomes, dtype=float64)     # Shape: 8 x BATCH_SIZE x SOL_DIMENSION

    # track time
    import time
    start = time.time()

    # optimize MES for each mutant batch & select best mutant
    new_genomes = np.empty((len(genomes), SOL_DIMENSION))
    new_acquisitions = np.empty((len(genomes), 1))
    for i in range(genomes.shape[0]):

        cellgrid = assamble_cellgrid(self, genomes_tensor[i])
        cellgrid = tensor(cellgrid, dtype=float64)      # Shape: 4000 x SOL_DIMENSION
        MES = qLowerBoundMaxValueEntropy(model=gp_model, candidate_set=cellgrid, num_mv_samples=NUM_MV_SAMPLES)

        new_genome, new_acquisition = optimize_acqf(
            acq_function=MES,
            bounds=tensor(cell_solutionbounds[i].T, dtype=float64),
            q=1,
            num_restarts=10,
            raw_samples=1024,
            #batch_initial_conditions=genomes_tensor[i].unsqueeze(0) 
        )

        new_genome = new_genome.detach().numpy()
        new_acquisition = new_acquisition.detach().numpy()

        new_genomes[i] = new_genome
        new_acquisitions[i] = new_acquisition

        perc_improvement = (((new_acquisition)/objectives[i]) - 1) * 100
        sum_perc_improvement += perc_improvement
        print("MES SAIL: {:.3f}  botorch.optimize_acqf(): {:.3f}  Improvement (Percent): {:.3f}".format(objectives[i], new_acquisition, perc_improvement))

    # Self.update_archive() will trigger acq_mes()
    # acq_mes() will calculate MES values for the new_genomes & their mutants
    # this will further optimize the MES values for the new_genomes
    self.update_archive(candidate_sol=new_genomes, candidate_bhv=new_genomes[:,1:3], acq_flag=True)

    mean_perc_improvement = sum_perc_improvement / n_samples
    print("Mean Improvement (Percent): ", mean_perc_improvement)
    print("Optimize MES Time: ", time.time() - start)

def simple_mes(self, genomes):

    print("\n\n\n\n\nentering simple_mes\n\n\n\n\n\n\n")
    # if genomes is empty, return empty array
    if len(genomes) == 0:
        return np.array([])

    genomes_tensor = tensor(genomes, dtype=float64)     # Shape: 8 x BATCH_SIZE x SOL_DIMENSION
    transformed_genomes = genomes_tensor.unsqueeze(1)   # Shape: 1 x BATCH_SIZE x 1 x SOL_DIMENSION

    acq_solution_tensor = tensor(np.zeros((len(genomes), SOL_DIMENSION)), dtype=float64)    # Shape: PARALLEL_BATCH_SIZE x 1
    acq_entropy_tensor = tensor(np.zeros((len(genomes), 1)), dtype=float64)                 # Shape: PARALLEL_BATCH_SIZE x 1
    for i in range(genomes.shape[0]):

        cellgrid = assamble_cellgrid(self, genomes_tensor[i,0])
        cellgrid = tensor(cellgrid, dtype=float64)      # Shape: 4000 x SOL_DIMENSION
        MES = qLowerBoundMaxValueEntropy(model=self.gp_model, candidate_set=cellgrid, num_y_samples=256)
        acq_entropy = MES(transformed_genomes[i].permute(1, 0, 2))

        elite_index = acq_entropy.argmax()
        acq_entropy_tensor[i] = acq_entropy[elite_index]
        acq_solution_tensor[i] = genomes_tensor[i,elite_index]

    # Store MES Elites in SailRunner class to use them inside the MAP-Loop
    mes_ndarray = acq_entropy_tensor.detach().numpy()

    return np.hstack(mes_ndarray)


def mes_sobol_cellgrids(self, mutant_cellrange, cell_indices=None):

    """
    In this module included only for documentation.
    The module is used within mes_map_elites.py

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

    Mutant Cellrange:
        Defines the boundaries where mutants are allowed to be sampled

        For example:
            mutant_cellrange = 0.1 -> mutants may exceed cellbounds by 10%
            mutant_cellrange = 0.0 -> mutants are not allowed to exceed cellbounds
            mutant_cellrange = -0.01 -> mutants are not allowed to exceed cellbounds, but are allowed to be sampled from a smaller cell

        Setting mutant_cellrange to -0.01 avoids edge cases

    Returns:

        n_acq_bins     : depends on archive resolution, may differ from n_obj_bins

        bhv_cellbounds : n_acq_bins x        2 dimensions       x   BHV_DIMENSION
        bhv_cellgrids  : n_acq_bins x   MES_GRID_SIZE samples   x   SOL_DIMENSION
        mes_cellgrid   :      1     x   MES_GRID_SIZE samples   x   SOL_DIMENSION

    # how does the naive approach work? : https://github.com/patrickab/thesis/blob/master/sail_xfoil/acq_functions/mes_cellgrid_documentation/MES%20Sobol%20Cellgrids.pdf
    # why would this approach be naive? : https://github.com/patrickab/thesis/blob/master/sail_xfoil/acq_functions/mes_cellgrid_documentation/MES%20Sobol%20Cellgrids.mp4

    """
    from chaospy import create_sobol_samples
    sobol_cellgrid = create_sobol_samples(order=MES_GRID_SIZE, dim=SOL_DIMENSION, seed=self.current_seed).T

    archive = self.acq_archive

    if cell_indices is None:
        n_cells = np.prod(archive.dims)
        archive_indices = range(n_cells)
        idx = archive.int_to_grid_index(archive_indices)
    else:
        n_cells = cell_indices.shape[0]
        idx = archive.int_to_grid_index(cell_indices)

    lower_bounds = np.array(SOL_VALUE_RANGE)[:, 0]
    upper_bounds = np.array(SOL_VALUE_RANGE)[:, 1]

    bhv_cellgrid = sobol_cellgrid[:, 1:3]
    mes_cellgrid = sobol_cellgrid * (upper_bounds - lower_bounds) + lower_bounds

    boundaries_0 = archive.boundaries[0]
    boundaries_1 = archive.boundaries[1]

    cell_range_0 = np.diff(boundaries_0)[0]
    cell_range_1 = np.diff(boundaries_1)[0]

    # 625 bins, MES_GRID_SIZE samples, 2 dimensions
    BHV_DIMENSION = 42
    bhv_cellgrids = np.empty((n_cells, MES_GRID_SIZE, BHV_DIMENSION))
    bhv_cellbounds = np.empty((n_cells, BHV_DIMENSION, 2))

    for i in range(n_cells):

        measure_0_idx, measure_1_idx = idx[i]

        # Allow mutants by scaling cellbounds
        cell_bounds_0 = (boundaries_0[measure_0_idx] - cell_range_0*mutant_cellrange,
                         boundaries_0[measure_0_idx+1] + cell_range_0*mutant_cellrange)

        cell_bounds_1 = (boundaries_1[measure_1_idx] - cell_range_1*mutant_cellrange,
                         boundaries_1[measure_1_idx+1] + cell_range_1*mutant_cellrange)

        # Restrict cellbounds to solution space boundaries
        cell_bounds_0 = np.clip(cell_bounds_0, boundaries_0[0], boundaries_0[-1])
        cell_bounds_1 = np.clip(cell_bounds_1, boundaries_1[0], boundaries_1[-1])

        cell_bounds_i = np.array([cell_bounds_0, cell_bounds_1])
        bhv_cellbounds[i] = cell_bounds_i

        lower_bounds = cell_bounds_i[:, 0]
        upper_bounds = cell_bounds_i[:, 1]
        cell_bound_ranges = upper_bounds - lower_bounds

        bhv_cellgrid_i = bhv_cellgrid.copy()        
        bhv_cellgrid_i = bhv_cellgrid_i * cell_bound_ranges.T + lower_bounds   # scale sobol cellgrid to cellbounds
        bhv_cellgrids[i] = bhv_cellgrid_i                                      # insert bhv cellgrid into mes cellgrid

    return bhv_cellbounds, bhv_cellgrids, mes_cellgrid

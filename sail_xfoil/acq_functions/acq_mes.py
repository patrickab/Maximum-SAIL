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

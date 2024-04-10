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
BATCH_SIZE = config.BATCH_SIZE


def acq_mes(self, genomes):

    # if genomes is empty, return empty array
    if len(genomes) == 0:
        return np.array([])

    genomes_tensor = tensor(genomes, dtype=float64)     # Shape: 8 x BATCH_SIZE x SOL_DIMENSION
    transformed_genomes = genomes_tensor.unsqueeze(1)   # Shape: 1 x BATCH_SIZE x 1 x SOL_DIMENSION

    acq_solution_tensor = tensor(np.zeros((len(genomes), SOL_DIMENSION)), dtype=float64)    # Shape: PARALLEL_BATCH_SIZE x 1
    acq_entropy_tensor = tensor(np.zeros((len(genomes), 1)), dtype=float64)                 # Shape: PARALLEL_BATCH_SIZE x 1
    for i in range(genomes.shape[0]):

        cellgrid = assamble_cellgrid(self, genomes_tensor[i])
        cellgrid = tensor(cellgrid, dtype=float64)      # Shape: 4000 x SOL_DIMENSION
        MES = qLowerBoundMaxValueEntropy(model=self.gp_model, candidate_set=cellgrid, num_mv_samples=NUM_MV_SAMPLES)
        acq_entropy = MES(transformed_genomes[i])
        
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

    mes_cellgrid = self.mes_sobol_cellgrid_mutants.copy()
    bhv_cellgrid_i = self.bhv_sobol_cellgrids_mutants[genome_cell_index].copy()

    mes_cellgrid[:,1:3] = bhv_cellgrid_i

    return mes_cellgrid
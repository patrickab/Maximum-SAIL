from ribs.archives import GridArchive
from numpy import float64
from chaospy import create_sobol_samples
import numpy as np
import os

### Custom Scripts ###
from utils.pprint_nd import pprint_nd, pprint
from utils.utils import scale_samples
from xfoil.generate_airfoils import generate_parsec_coordinates
from xfoil.simulate_airfoils import xfoil

### Configurable Variables ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
INIT_N_EVALS = config.INIT_N_EVALS
SOL_DIMENSION = config.SOL_DIMENSION
BATCH_SIZE = config.BATCH_SIZE
OBJ_DIMENSION = config.OBJ_DIMENSION
BHV_DIMENSION = config.BHV_DIMENSION
TEST_RUNS = config.TEST_RUNS

# xfoil parameters
ALFA = config.ALFA
MACH = config.MACH
REYNOLDS = config.REYNOLDS

def initialize_archive(archive: GridArchive, seed: int):

    print("\nInitialize init_archive() [...]")

    init_solutions = np.empty((0, SOL_DIMENSION), dtype=float64)
    init_obj_evals = np.empty((0, OBJ_DIMENSION), dtype=float64)
    init_bhv_evals = np.empty((0, BHV_DIMENSION), dtype=float64)

    n_evals = INIT_N_EVALS
    
    while n_evals >= BATCH_SIZE:

        n_evals -= BATCH_SIZE

        solutions = np.empty((0, SOL_DIMENSION), dtype=float64)
        bhv_evals = np.empty((0, BHV_DIMENSION), dtype=float64)
        
        samples = create_sobol_samples(order=BATCH_SIZE, dim=len(SOL_VALUE_RANGE), seed=seed)
        samples = samples.T

        seed += TEST_RUNS
        
        scale_samples(samples)                  # sobol samples produce values between [0;1]

        valid_indices, surface_area_batch = generate_parsec_coordinates(samples)
        convergence_errors, success_indices, obj_batch = xfoil(iterations=BATCH_SIZE)

        if convergence_errors == BATCH_SIZE:
            continue

        for index in success_indices:

            solutions = np.vstack((solutions, samples[index]))                        # Store converged solution
            bhv_eval = np.array([samples[index][1], samples[index][2]])               # Calculate behavior
            
            bhv_evals = np.vstack((bhv_evals, bhv_eval))                              # Store behavior for archive

            init_solutions = np.vstack((init_solutions, samples[index]))              # Store init solutions for GP
            init_bhv_evals = np.vstack((init_bhv_evals, bhv_eval))                    # Store init behavior for GP

        init_obj_evals = np.vstack((init_obj_evals, obj_batch.reshape(-1,1)))         # Store init objective for GP

        print("Current Elites in Archive (before): " + str(archive.stats.num_elites))
        archive.add(solutions, obj_batch, bhv_evals)
        print("Current Elites in Archive (after): " + str(archive.stats.num_elites))
    
    print("[...] Terminate init_archive()\n")
    return archive, init_solutions, init_obj_evals
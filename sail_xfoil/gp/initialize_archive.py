from ribs.archives import GridArchive
from numpy import float64
from chaospy import create_sobol_samples
import dask.array as da
import numpy as np
import os

### Custom Scripts ###
from utils.pprint import pprint
from utils.pprint_nd import pprint_nd
from utils.scale_samples import scale_samples
from xfoil.generate_airfoils import generate_parsec_coordinates

from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
INIT_ARCHIVE_SIZE = config.INIT_ARCHIVE_SIZE
SOL_DIMENSION = config.SOL_DIMENSION
PARALLEL_BATCH_SIZE = config.PARALLEL_BATCH_SIZE
OBJ_DIMENSION = config.OBJ_DIMENSION
BHV_DIMENSION = config.BHV_DIMENSION

def initialize_archive(archive: GridArchive, fuct_obj, fuct_bhv):

    print("Initialize init_archive() [...]\n")

    init_solutions = np.empty((0, SOL_DIMENSION), dtype=float64)
    init_obj_evals = np.empty((0, OBJ_DIMENSION), dtype=float64)
    init_bhv_evals = np.empty((0, BHV_DIMENSION), dtype=float64)

    i_seed = 123

    while archive.stats.num_elites < INIT_ARCHIVE_SIZE:

        n_missing = (INIT_ARCHIVE_SIZE - archive.stats.num_elites)
        n_samples = PARALLEL_BATCH_SIZE if (n_missing>PARALLEL_BATCH_SIZE) else n_missing

        samples = create_sobol_samples(order=n_samples, dim=len(SOL_VALUE_RANGE), seed=i_seed)
        samples = samples.T
        
        scale_samples(samples)
        
        generate_parsec_coordinates(samples)

        i_seed = (i_seed + 321) % 1234

        obj_evals = fuct_obj(samples)       # Calculate objective
        bhv_evals = fuct_bhv(samples)       # Calculate behavior

        print("\nAdd samples to Archive")
        print("Current Elites in Archive (before): " + str(archive.stats.num_elites))
        archive.add(samples, obj_evals, bhv_evals)            # Store elite solutions
        print("Current Elites in Archive  (after): " + str(archive.stats.num_elites))

        init_solutions = np.vstack((init_solutions, samples))
        init_obj_evals = np.vstack((init_obj_evals, obj_evals.reshape(-1,1)))
        init_bhv_evals = np.vstack((init_bhv_evals, bhv_evals))

        if archive.stats.num_elites != INIT_ARCHIVE_SIZE:
            print("")
    

    print("\n[...] Terminate init_archive()\n")
    return archive, init_solutions, init_obj_evals, init_bhv_evals 
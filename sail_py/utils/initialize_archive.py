from ribs.archives import GridArchive
from chaospy import create_sobol_samples
import numpy
import os

### Custom Scripts ###
from utils.pprint import pprint

from config import Config
config = Config(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
INIT_ARCHIVE_SIZE = config.INIT_ARCHIVE_SIZE
SOL_DIMENSION = config.SOL_DIMENSION
PARALLEL_BATCH_SIZE = config.PARALLEL_BATCH_SIZE
OBJ_DIMENSION = config.OBJ_DIMENSION


def initialize_archive(archive: GridArchive, example_objective_function, example_behavior_function):

    print("\nInitialize init_archive() [...]\n")

    init_solutions = numpy.empty((0, SOL_DIMENSION), dtype=float)
    init_obj_evals = numpy.empty((0, OBJ_DIMENSION), dtype=float)
    i_seed = 123

    while archive.stats.num_elites < INIT_ARCHIVE_SIZE:

        n_missing = (INIT_ARCHIVE_SIZE - archive.stats.num_elites)
        n_samples = PARALLEL_BATCH_SIZE if (n_missing>PARALLEL_BATCH_SIZE) else n_missing

        samples = create_sobol_samples(order=n_samples, dim=len(SOL_VALUE_RANGE), seed=i_seed)
        samples = samples.T

        for i in range(len(samples)):
            for j in range(len(samples[i])):
                lower_bound, upper_bound = SOL_VALUE_RANGE[j]
                samples[i][j] = samples[i][j] *(upper_bound - lower_bound) + lower_bound

        i_seed = (i_seed + 321) % 2000

        obj_eval = example_objective_function(samples)      # Calculate objective
        bhv_eval = example_behavior_function(samples)       # Calculate behavior

        print("Add samples to Archive")
        print("Current Elites in Archive: " + str(archive.stats.num_elites))
        archive.add(samples, obj_eval, bhv_eval)            # Store elite solutions
        print("Current Elites in Archive: " + str(archive.stats.num_elites))

        init_solutions = numpy.vstack((init_solutions, samples))
        init_obj_evals = numpy.vstack((init_obj_evals, obj_eval.reshape(-1,1)))

        if archive.stats.num_elites != INIT_ARCHIVE_SIZE:
            print("")

    print("\n[...] Terminate init_archive()\n")
    return archive, init_solutions, init_obj_evals
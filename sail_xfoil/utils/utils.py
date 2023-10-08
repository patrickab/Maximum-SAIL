import os
import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter

### Global Parameters ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SOL_DIMENSION = config.SOL_DIMENSION
BHV_VALUE_RANGE = config.BHV_VALUE_RANGE
BHV_NUMBER_BINS = config.BHV_NUMBER_BINS
BATCH_SIZE = config.BATCH_SIZE
SIGMA_EMITTER = config.SIGMA_EMITTER


# elite_status_vector = archive.add(acq_elites, obj_evals, bhv_evals)
        # elite_status_vector == 0  ->  acq_elite was not added
        # elite_status_vector == 1  ->  acq_elite was added
        # elite_status_vector == 2  ->  acq_elite discovered new cell
def select_new_elites(elite_status_vector,candidate_elites, obj_evals):

    x_new_elites = []
    obj_new_elites = []

    for candidate_elite, obj_eval ,elite_status_value in candidate_elites, obj_evals, elite_status_vector:
        if elite_status_value > 0:
            x_new_elites.append(candidate_elite)
            obj_new_elites.append(obj_eval)

    return x_new_elites, obj_new_elites

#def count_new_elites(elite_status_vector):


def scale_samples(samples):
    for i in range(len(samples)):
        for j in range(len(samples[i])):
            lower_bound, upper_bound = SOL_VALUE_RANGE[j]
            samples[i][j] = samples[i][j] *(upper_bound - lower_bound) + lower_bound

    return samples


def define_archives(initial_seed):

    obj_archive = GridArchive(
        solution_dim=SOL_DIMENSION,         # Dimension of solution vector
        dims=BHV_NUMBER_BINS,               # Discretization of behavioral bins
        ranges=BHV_VALUE_RANGE,             # Possible values for behavior vector
        qd_score_offset=-600,
        threshold_min = -1,
        seed=initial_seed
        )
    
    acq_archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = -1,
        seed=initial_seed
        )
    
    pred_archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = -1,
        seed=initial_seed,
        )
    
    return obj_archive, acq_archive, pred_archive


def define_emitter(init_solutions, archive, seed, sol_value_range=None):

    if sol_value_range is None:
        sol_value_range = SOL_VALUE_RANGE

    emitter = [
        GaussianEmitter(
        archive=archive,
        sigma=SIGMA_EMITTER,
        bounds= np.array(sol_value_range),
        batch_size=BATCH_SIZE,
        initial_solutions=init_solutions,
        seed=seed
    )]

    return emitter

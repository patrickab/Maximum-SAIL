import os
import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter

### Custom Scripts ###
from xfoil.simulate_airfoils import xfoil
from xfoil.generate_airfoils import generate_parsec_coordinates

### Global Parameters ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SOL_DIMENSION = config.SOL_DIMENSION
BHV_VALUE_RANGE = config.BHV_VALUE_RANGE
BHV_NUMBER_BINS = config.BHV_NUMBER_BINS
BATCH_SIZE = config.BATCH_SIZE
SIGMA_EMITTER = config.SIGMA_EMITTER
SIGMA_PRED_EMITTER = config.SIGMA_PRED_EMITTER


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


def generate_emitter(init_solutions, archive, seed, sigma_emitter=SIGMA_EMITTER, sol_value_range=None):

    if sol_value_range is None:
        sol_value_range = SOL_VALUE_RANGE

    emitter = [
        GaussianEmitter(
        archive=archive,
        sigma=sigma_emitter,
        bounds= np.array(sol_value_range),
        batch_size=BATCH_SIZE,
        initial_solutions=init_solutions,
        seed=seed
    )]

    return emitter


def eval_xfoil_loop(samples, behavior):
    """
    XFOIL evaluation is performed in Batches of BATCH_SIZE
        Therefore, if n_samples != BATCH_SIZE, 
        samples need to be evaluated in a loop

    input:
        samples     Type: ndarrayn_samples      Shape: (n_samples, SOL_DIMENSION)
    """
    conv_sol = np.empty(0)
    conv_obj = np.empty(0)
    conv_bhv = np.empty(0)
        
    for index in range(0 ,samples.shape[0], BATCH_SIZE):
        generate_parsec_coordinates(samples[index:index+BATCH_SIZE])

        n_solutions = len(samples[index:BATCH_SIZE])
        _, success_indices, new_elite_objectives = xfoil(n_solutions)

        converged_sol = samples[index:BATCH_SIZE][success_indices]
        converged_bhv = behavior[index:BATCH_SIZE][success_indices]

        if converged_sol.shape[0] != 0: # if converged_sol is not empty
            conv_sol = np.concatenate(conv_sol, converged_sol) if conv_sol.size else converged_sol # if conv_sol is empty, initialize with converged_sol
            conv_obj = np.concatenate(conv_obj, new_elite_objectives) if conv_obj.size else new_elite_objectives
            conv_bhv = np.concatenate(conv_bhv, converged_bhv) if conv_bhv.size else converged_bhv

    return conv_sol, conv_obj, conv_bhv


def init_pred_archive(pred_archive, obj_archive, seed, sigma_emitter=SIGMA_PRED_EMITTER):
    pred_archive.add([elite.solution for elite in obj_archive], [elite.objective for elite in obj_archive], [elite.measures for elite in obj_archive])
    pred_emitter = generate_emitter(init_solutions=[elite.solution for elite in obj_archive], archive=pred_archive, seed=seed, sigma_emitter=sigma_emitter)
    return pred_archive, pred_emitter
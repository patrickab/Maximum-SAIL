import os
import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter

### Custom Scripts ###
from xfoil.generate_airfoils import generate_parsec_coordinates
from xfoil.simulate_airfoils import xfoil
from gp import predict_objective
from acq_functions.acq_ucb import acq_ucb
from utils.pprint_nd import pprint

### Global Parameters ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
BATCH_SIZE = config.BATCH_SIZE
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SIGMA_EMITTER = config.SIGMA_EMITTER


def eval_xfoil_loop(self, candidate_sol, pred_flag, acq_flag):
    """
    Ensures that iter_samples <= BATCH_SIZE are evaluated
    
    XFOIL evaluation is performed in Batches of BATCH_SIZE
        Therefore, if n_samples != BATCH_SIZE, 
        samples need to be evaluated in a loop

    Input:
        samples     Type: ndarrayn_samples      Shape: (n_samples, SOL_DIMENSION)
    returns:
        conv_sol_batch, conv_obj_batch, conv_bhv_batch, archive, extra_evals

    """

    n_errors = 0
    iteration = 0
    remaining_samples = candidate_sol.shape[0]

    while remaining_samples>0: # allows indices [0:10], [10:20], [20:22]

        sample_index = iteration*BATCH_SIZE
        i_solutions = candidate_sol[sample_index:sample_index+BATCH_SIZE] # alows indices eg [20:22] to be sampled, if 2 samples are left
        iteration += 1

        if i_solutions.shape[0] > BATCH_SIZE:
            raise ValueError(f'eval_xfoil_loop: i_solutions.shape[0] > BATCH_SIZE')
        
        i_candidates = i_solutions.shape[0]
        remaining_samples -= i_candidates

        # generate .dat files
        i_solutions = np.vstack(i_solutions)
        generate_parsec_coordinates(i_solutions)

        # evaluate samples batch & extract converged solutions
        _, success_indices, converged_obj = xfoil(i_candidates)
        success_indices = success_indices[:i_candidates]
        converged_sol = i_solutions[success_indices]
        converged_bhv = i_solutions[:,1:3][success_indices] # ToDo: generalize calculate behavior

        i_errors = i_candidates - len(success_indices)
        n_errors += i_errors

        if i_errors < 0:
            raise ValueError(f'eval_xfoil_loop: i_errors < 0')
        
        # add converged solutions & render .pngs - if specified update & render other archive(s)
        self.update_archive(candidate_sol=converged_sol, candidate_obj=converged_obj, candidate_bhv=converged_bhv, obj_flag=True, pred_flag=pred_flag, acq_flag=acq_flag)

        # iteratively store new data data in self._sol_array & self._obj_array
        self.update_gp_data(new_solutions=converged_sol, new_objectives=converged_obj)

    # update GP model with new data
    self.set_n_errors(n_errors)
    self.update_gp_model()
    return


def store_n_best_elites(archive: GridArchive, n: int, update_acq=True, gp_model=None, obj_archive=None):
    """
    Store best elites from an archive

        options:   - store n best elites from archive in archive
                   - store elites from both archives in acq_archive
                   - update acquisition values of acq_archive
    """

    if obj_archive is None:
        obj_archive = archive

    # ToDO: generalize variable names

    n_obj_elites = sorted(obj_archive, key=lambda x: x.objective, reverse=True)[:n]
    n_acq_elites = sorted(archive, key=lambda x: x.objective, reverse=True)[:n]

    n_obj_sol = np.array([elite.solution for elite in n_obj_elites])
    n_acq_sol = np.array([elite.solution for elite in n_acq_elites])

    if update_acq:
        n_obj_elite_acq = acq_ucb(n_obj_sol, gp_model)
        n_acq_elite_acq = acq_ucb(n_acq_sol, gp_model)
    else:
        n_obj_elite_acq = np.array([elite.objective for elite in n_obj_elites])
        n_acq_elite_acq = np.array([elite.objective for elite in n_acq_elites])

    n_sol = np.concatenate((n_obj_sol, n_acq_sol), axis=0)
    n_acq = np.concatenate((n_obj_elite_acq, n_acq_elite_acq), axis=0)
    n_bhv = np.concatenate(([elite.measures for elite in n_obj_elites], [elite.measures for elite in n_acq_elites]), axis=0)

    archive.clear()
    archive.add(n_sol, n_acq, n_bhv)

    return archive


def scale_samples(samples, boundaries=SOL_VALUE_RANGE):
    """Scales Samples to boundaries"""

    # ToDo: vectorize
    for i in range(len(samples)):
        for j in range(len(samples[i])):
            lower_bound, upper_bound = boundaries[j]
            samples[i][j] = samples[i][j] *(upper_bound - lower_bound) + lower_bound

    return samples


def generate_emitter(init_solutions, archive, seed, sigma_emitter=SIGMA_EMITTER, sol_value_range=None):
    """Reduces Overhead"""

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


def calculate_behavior(solutions):
    if solutions.size == 0:
        return np.array([])
    elif solutions.shape[0] == 1:
        return solutions[0][1:3]
    else:
        return solutions[:][1:3]

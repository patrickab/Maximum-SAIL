import os
import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter

### Custom Scripts ###
from xfoil.generate_airfoils import generate_parsec_coordinates
from xfoil.simulate_airfoils import xfoil
from acq_functions.acq_ucb import acq_ucb
from utils.pprint_nd import pprint

### Global Parameters ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
BATCH_SIZE = config.BATCH_SIZE
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SIGMA_EMITTER = config.SIGMA_EMITTER


def eval_xfoil_loop(self, candidate_sol, candidate_bhv, obj_flag=False, pred_flag=False, acq_flag=False, archive=None):
    """
    XFOIL evaluation is performed in Batches of BATCH_SIZE
        Therefore, if n_samples != BATCH_SIZE, 
        samples need to be evaluated in a loop

    Always ensure to call self.update_gp_model() after calling this function
        This feature is not integrated, in order to allow for more flexibility

    - ensures that iter_samples <= BATCH_SIZE are evaluated

    input:
        samples     Type: ndarrayn_samples      Shape: (n_samples, SOL_DIMENSION)
    returns:
        conv_sol_batch, conv_obj_batch, conv_bhv_batch, archive, extra_evals

    """
    conv_sol_batch = np.empty(0)
    conv_obj_batch = np.empty(0)
    conv_bhv_batch = np.empty(0)
    succes_indices_batch = np.empty(0)


    iteration = 0
    n_candidates=candidate_sol.shape[0]
    remaining_samples = n_candidates

    while remaining_samples>0: # allows indices [0:10], [10:20], [20:22]

        sample_index = iteration*BATCH_SIZE
        i_solutions = candidate_sol[sample_index:sample_index+BATCH_SIZE] # alows indices eg [20:22] to be sampled, if 2 samples are left
        i_behaviors = candidate_bhv[sample_index:sample_index+BATCH_SIZE]
        iteration += 1
        
        i_candidates = i_solutions.shape[0]
        remaining_samples -= i_candidates

        # generate .dat files
        i_solutions = np.vstack(i_solutions)
        generate_parsec_coordinates(i_solutions)

        # evaluate samples batch & extract converged solutions
        _, success_indices, converged_obj = xfoil(i_candidates)
        success_indices = success_indices[:i_candidates]
        converged_sol = i_solutions[success_indices]
        converged_bhv = i_behaviors[success_indices]
        success_indices = np.hstack(np.vstack(success_indices) + sample_index)
        succes_indices_batch = np.vstack((succes_indices_batch, success_indices*iteration*BATCH_SIZE)) if succes_indices_batch.size > 0 else success_indices

        # add converged solutions & render .pngs - if specified update & render other archive(s)
        self.update_archive(candidate_sol=converged_sol, candidate_obj=converged_obj, candidate_bhv=converged_bhv, obj_flag=True)
        if pred_flag:
            self.update_archive(candidate_sol=converged_sol, candidate_obj=converged_obj, candidate_bhv=converged_bhv, pred_flag=True)
        if acq_flag:
            self.update_archive(candidate_sol=converged_sol, candidate_obj=converged_obj, candidate_bhv=converged_bhv, acq_flag=True)

        # prepare data for calling function
        if converged_sol.shape[0] != 0:
            if conv_sol_batch.size > 0:
                conv_sol_batch = np.vstack((conv_sol_batch, converged_sol))
                conv_obj_batch = np.append(conv_obj_batch, converged_obj)
                conv_bhv_batch = np.vstack((conv_bhv_batch, converged_bhv))
            else:
                conv_sol_batch = converged_sol
                conv_obj_batch = converged_obj
                conv_bhv_batch = converged_bhv
    return conv_sol_batch, conv_obj_batch, conv_bhv_batch, succes_indices_batch, archive


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

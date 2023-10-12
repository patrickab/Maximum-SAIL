import os
import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter

### Custom Scripts ###
from xfoil.generate_airfoils import generate_parsec_coordinates
from xfoil.simulate_airfoils import xfoil
from acq_functions.acq_ucb import acq_ucb
from utils.pprint_nd import pprint_nd, pprint, pprint_fstring

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
MAX_PRED_VERIFICATION = config.MAX_PRED_VERIFICATION
PRED_ELITE_REEVALS = config.PRED_ELITE_REEVALS


def eval_xfoil_loop(self, samples, behavior, extra_evals, obj_flag=False, pred_flag=False, acq_flag=False, archive=None):
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
    extra_evals = 0
    total_samples=samples.shape[0]
    remaining_samples = total_samples

    while remaining_samples>0: # allows indices [0:10], [10:20], [20:22]

        sample_index = iteration*BATCH_SIZE
        iteration_sols = samples[sample_index:sample_index+BATCH_SIZE] # alows indices eg [20:22] to be sampled, if 2 samples are left
        iteration_bhvs = behavior[sample_index:sample_index+BATCH_SIZE]
        iteration += 1
        
        n_samples = iteration_sols.shape[0]
        remaining_samples -= n_samples
        
        if pred_flag:
            extra_evals += n_samples

        iteration_sols = np.vstack(iteration_sols)
        generate_parsec_coordinates(iteration_sols)

        _, success_indices, converged_obj = xfoil(n_samples) # ToDo: modify xfoil to take in sample sizes below BATCH_SIZE
        success_indices = success_indices[:n_samples]

        converged_sol = iteration_sols[success_indices]
        converged_bhv = iteration_bhvs[success_indices]

        success_indices = np.hstack(np.vstack(success_indices) + sample_index)
        succes_indices_batch = np.vstack((succes_indices_batch, success_indices*iteration*BATCH_SIZE)) if succes_indices_batch.size > 0 else success_indices

        self.update_archive(converged_sol, converged_obj, converged_bhv, obj_flag=True)
        if pred_flag:
            self.update_archive(converged_sol, converged_obj, converged_bhv, pred_flag=True)
        if acq_flag:
            self.update_archive(converged_sol, converged_obj, converged_bhv, acq_flag=True)

        if converged_sol.shape[0] != 0: # if converged_sol is not empty
            if conv_sol_batch.size > 0:
                conv_sol_batch = np.vstack((conv_sol_batch, converged_sol))
                conv_obj_batch = np.append(conv_obj_batch, converged_obj)
                conv_bhv_batch = np.vstack((conv_bhv_batch, converged_bhv))
            else:
                conv_sol_batch = converged_sol
                conv_obj_batch = converged_obj
                conv_bhv_batch = converged_bhv

    return conv_sol_batch, conv_obj_batch, conv_bhv_batch, succes_indices_batch, archive, extra_evals


def maximize_obj_improvement(new_elite_archive: GridArchive, old_elites: np.ndarray):
    """
    - extracts all elites from new_elite_archive
    - orders them by objective improvement

    Input: 
        (Grid_Archive): new_elite_archive
        (np_ndarray): old elites         
            -> old_elites = np.array([(elite.solution, elite.index, elite.objective, elite.measures) for elite in obj_archive], dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)]))
    """
    # ToDo: Verify

    elites = np.array([(elite.solution, elite.index, elite.objective, elite.measures) for elite in new_elite_archive], dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])
    elites = elites[np.argsort(elites['index'])]
    print(elites)
    
    # Seperate improved elites (niche compete) from new elites (new niches)
    is_improved_new_elite = np.isin(elites['index'], old_elites['index'])
    improved_elites = elites[is_improved_new_elite]
    new_elites      = elites[~is_improved_new_elite]

    # Sort by index
    improved_elites = improved_elites[np.argsort(improved_elites['index'])]
    new_elites      = new_elites[np.argsort(new_elites['index'])]

    # Select old elites that have been improved
    is_improved_old_elite = np.isin(old_elites['index'], improved_elites['index'])
    old_elites_improved   = old_elites[is_improved_old_elite]
    print(old_elites_improved)

    objective_improvement = improved_elites['objective'] - old_elites_improved['objective']

    # Pack into one data structure
    improved_elites = np.array(list(zip(
        improved_elites['solution'], improved_elites['objective'],          objective_improvement , improved_elites['behavior'])), 
        dtype=[        ('solution', object),        ('objective', float), ('objective_improvement', float),        ('behavior', object)])
    # Sort & flip to ensure descending order
    improved_elites = improved_elites[np.argsort(improved_elites['objective_improvement'])]
    improved_elites = np.flip(improved_elites)
    
    new_elites = np.array(list(zip(
        new_elites['solution'], new_elites['objective'],        new_elites['objective'], new_elites['behavior'])), 
           dtype=[('solution', object),   ('objective', float),('objective_improvement', float),   ('behavior', object)])
    
    new_elites      = new_elites[np.argsort(new_elites['objective_improvement'])]
    new_elites      = np.flip(new_elites)

    n_obj_improvements = improved_elites.shape[0] + new_elites.shape[0]

    return improved_elites, new_elites, n_obj_improvements


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



def select_new_elites(elite_status_vector,candidate_elites, obj_evals):

    x_new_elites = []
    obj_new_elites = []

    for candidate_elite, obj_eval ,elite_status_value in candidate_elites, obj_evals, elite_status_vector:
        if elite_status_value > 0:
            x_new_elites.append(candidate_elite)
            obj_new_elites.append(obj_eval)

    return x_new_elites, obj_new_elites


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

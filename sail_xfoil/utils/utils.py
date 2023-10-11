import os
import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter

### Custom Scripts ###
from utils.anytime_archive_visualizer import anytime_archive_visualizer
from xfoil.generate_airfoils import generate_parsec_coordinates
from xfoil.simulate_airfoils import xfoil
from acq_functions.acq_ucb import acq_ucb

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


def eval_xfoil_loop(samples, behavior, # arguments below used for anytime archive visualizer
                    extra_evals, archive=None, benchmark_domain=None, initial_seed=None, index_anytime_visualizer=None):
    """
    XFOIL evaluation is performed in Batches of BATCH_SIZE
        Therefore, if n_samples != BATCH_SIZE, 
        samples need to be evaluated in a loop

    - ensures that batches of EXACTLY BATCH_SIZE are evaluated

    input:
        samples     Type: ndarrayn_samples      Shape: (n_samples, SOL_DIMENSION)
    returns:
        conv_sol, conv_obj, conv_bhv, archive, extra_evals

    """
    conv_sol = np.empty(0)
    conv_obj = np.empty(0)
    conv_bhv = np.empty(0)

    index_visualizer=index_anytime_visualizer
    total_samples=samples.shape[0]
    remaining_samples = total_samples

    for index in range(0, total_samples, 10): # allows indices [0:10], [10:20], [20:22]

        sample_index = index*BATCH_SIZE
        iteration_sols = samples[sample_index:sample_index+BATCH_SIZE] # alows indices eg [20:22] to be sampled, if 2 samples are left
        iteration_bhvs = behavior[sample_index:sample_index+BATCH_SIZE] 
        n_samples = iteration_sols.shape[0]
        remaining_samples -= n_samples

        generate_parsec_coordinates(iteration_sols)

        _, success_indices, new_elite_objectives = xfoil(n_samples) # ToDo: modify xfoil to take in sample sizes below BATCH_SIZE
        success_indices = success_indices[:n_samples]
        extra_evals += n_samples

        converged_sol = iteration_sols[success_indices]
        converged_bhv = iteration_bhvs[success_indices]

        print("Eval Xfoil Loop Elites (before):  " + str(archive.stats.num_elites))
        archive.add(converged_sol, new_elite_objectives, converged_bhv)
        print("Eval Xfoil Loop Elites (after): " + str(archive.stats.num_elites))

        anytime_archive_visualizer(archive, benchmark_domain, initial_seed, index_visualizer)
        index_visualizer += 1
        print(index_visualizer)

        if converged_sol.shape[0] != 0: # if converged_sol is not empty
            if conv_sol.size > 0:
                conv_sol = np.vstack((conv_sol, converged_sol))
                conv_obj = np.append(conv_obj, new_elite_objectives)
                conv_bhv = np.vstack((conv_bhv, converged_bhv))
            else:
                conv_sol = converged_sol
                conv_obj = new_elite_objectives
                conv_bhv = converged_bhv

    return conv_sol, conv_obj, conv_bhv, archive, extra_evals


def maximize_obj_improvement(new_elite_archive: GridArchive, old_elites: np.ndarray):
    """
    - extracts all elites from new_elite_archive
    - orders them by objective improvement

    Input: 
        (Grid_Archive): new_elite_archive
        (np_ndarray): old elites         
            -> old_elites = np.array([(elite.solution, elite.index, elite.objective, elite.measures) for elite in obj_archive], dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)]))
    """

    
    new_elites = np.array(
        [(elite.solution, elite.index, elite.objective, elite.measures) for elite in new_elite_archive], 
        dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])
    
    is_improved_mask = np.isin(new_elites['index'], old_elites['index'])

    # seperate improved elites (niche compete) from new elites (new niches)
    improved_elites = new_elites[is_improved_mask]
    improved_elites = improved_elites[np.argsort(improved_elites['index'])]
    new_elites = new_elites[~is_improved_mask]

    # select old elites that are in improved elites
    is_old_in_improved_mask = np.isin(old_elites['index'], improved_elites['index'])
    old_elites_improved = old_elites[is_old_in_improved_mask]

    max_acq_improvement_elites = np.array(list(zip(
        improved_elites['solution'], 
        improved_elites['objective'],
        (improved_elites['objective'] - old_elites_improved['objective']), 
        improved_elites['behavior'])), 
        dtype=[('solution', object), ('objective', float), ('objective_improvement', float), ('behavior', object)])
    
    max_acq_improvement_elites = max_acq_improvement_elites[np.argsort(max_acq_improvement_elites['objective_improvement'])]
    max_acq_improvement_elites = np.flip(max_acq_improvement_elites)
    new_elites = np.array(list(zip(new_elites['solution'], new_elites['objective'], new_elites['objective'], new_elites['behavior'])), 
                                      dtype=[('solution', object), ('objective', float),('objective_improvement', float), ('behavior', object)])

    return max_acq_improvement_elites, new_elites


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


def init_pred_archive(pred_archive, obj_archive, seed, sigma_emitter=SIGMA_PRED_EMITTER):
    """
    - Stores Obj Elites in Pred Archive
    - Generates Emitter for Pred Archive
    """
    pred_archive.add([elite.solution for elite in obj_archive], [elite.objective for elite in obj_archive], [elite.measures for elite in obj_archive])
    pred_emitter = generate_emitter(init_solutions=[elite.solution for elite in obj_archive], archive=pred_archive, seed=seed, sigma_emitter=sigma_emitter)
    return pred_archive, pred_emitter


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


def define_archives(initial_seed):
    """Reduces Overhead"""

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
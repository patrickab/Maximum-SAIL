import os
import gc
import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter

### Custom Scripts ###
from xfoil.generate_airfoils import generate_parsec_coordinates
from xfoil.simulate_airfoils import xfoil
from gp.predict_objective import predict_objective
from acq_functions.acq_ucb import acq_ucb
from utils.pprint_nd import pprint

### Global Parameters ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
BATCH_SIZE = config.BATCH_SIZE
SIGMA_EMITTER = config.SIGMA_EMITTER
SOL_DIMENSION = config.SOL_DIMENSION
BHV_DIMENSION = config.BHV_DIMENSION
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE


def eval_xfoil_loop(self, candidate_sol, evaluate_prediction_archive=False, candidate_acq=None):
    """
    Ensures that iter_samples <= BATCH_SIZE are evaluated
    
    XFOIL evaluation is performed in Batches of BATCH_SIZE
        Therefore, if n_samples != BATCH_SIZE, 
        samples need to be evaluated in a loop

    Obj Elites are always stored & rendered in their archive
    Subsequently the GP model is updated with new data

    Finally, the acq/pred archives are updated with obj elites & new gp model
    """

    self.obj_t0 = self.obj_archive.stats.num_elites
    n_errors = 0
    iteration = 0
    n_new_sol = np.array([])
    new_solutions = np.empty((0, SOL_DIMENSION))
    new_behaviors = np.empty((0, BHV_DIMENSION))
    remaining_samples = candidate_sol.shape[0]

    old_obj_elites = sorted(self.obj_archive, key=lambda x: x.objective, reverse=True)[:self.obj_archive.stats.num_elites]
    old_obj_solutions = np.array([elite.solution for elite in old_obj_elites])
    old_obj_behavior = np.array([elite.measures for elite in old_obj_elites])

    old_acq_elites = sorted(self.acq_archive, key=lambda x: x.objective, reverse=True)[:self.acq_archive.stats.num_elites]
    old_acq_solutions = np.array([elite.solution for elite in old_acq_elites])
    old_acq_behavior = np.array([elite.measures for elite in old_acq_elites])

    old_pred_elites = sorted(self.pred_archive, key=lambda x: x.objective, reverse=True)[:self.pred_archive.stats.num_elites]
    old_pred_solutions = np.array([elite.solution for elite in old_pred_elites])
    old_pred_behavior = np.array([elite.measures for elite in old_pred_elites])

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

        if candidate_acq is not None:
            converged_acq = candidate_acq[success_indices]
            obj = converged_obj[converged_obj >= converged_acq]
            acq = converged_acq[converged_obj >= converged_acq]
            pprint(converged_obj, converged_acq)

        if i_errors < 0:
            raise ValueError(f'eval_xfoil_loop: i_errors < 0')
        
        # iteratively store new data
        if not evaluate_prediction_archive:
            self.update_gp_data(new_solutions=converged_sol, new_objectives=converged_obj)
            self.update_archive(candidate_sol=converged_sol, candidate_obj=converged_obj, candidate_bhv=converged_bhv, obj_flag=True)

            self.visualize_archive(self.new_archive, new_flag=True)
            self.visualize_archive(self.obj_archive, obj_flag=True)
        else:
            self.update_archive(candidate_sol=converged_sol, candidate_obj=converged_obj, candidate_bhv=converged_bhv, evaluate_prediction_archive=True)

        # --- All following steps are only performed in this manner to ensure that all rendered videos are of equal length ---
        # ---        New elites are iteratively stored to later iteratively update archives with up2date gp model          ---
        i_new_sol = converged_obj.shape[0]
        n_new_sol = np.array(np.append(n_new_sol, i_new_sol), dtype=int)


        new_sol = converged_sol
        new_solutions = np.vstack((new_solutions, new_sol))

        new_bhv = converged_bhv
        new_behaviors = np.vstack((new_behaviors, new_bhv))

    # evaluate candidates, then exit loop
    if evaluate_prediction_archive:
        return

    self.update_gp_model()

    # update GP model with new data
    self.acq_archive.clear()
    self.pred_archive.clear()

    # Combine old_obj_solutions and old_(pred/acq)_solutions, then remove duplicates
    old_acq_sol = np.concatenate((old_obj_solutions, old_acq_solutions), axis=0)
    old_pred_sol = np.concatenate((old_obj_solutions, old_pred_solutions), axis=0)

    old_acq_bhv = np.concatenate((old_obj_behavior, old_acq_behavior), axis=0)
    old_pred_bhv = np.concatenate((old_obj_behavior, old_pred_behavior), axis=0)

    if old_acq_sol.shape[0] != old_acq_bhv.shape[0] or old_pred_sol.shape[0] != old_pred_bhv.shape[0]:
        raise ValueError(f'old_acq_solutions.shape[0] != old_acq_behavior.shape[0] or old_pred_solutions.shape[0] != old_pred_behavior.shape[0]')

    self.update_archive(candidate_sol=old_acq_solutions, candidate_bhv=old_acq_behavior, acq_flag=True)
    self.update_archive(candidate_sol=old_pred_solutions, candidate_bhv=old_pred_behavior, pred_flag=True)

    sum = 0
    for i in range(n_new_sol.shape[0]):
        new_sol = new_solutions[sum:sum+n_new_sol[i]]
        new_bhv = new_behaviors[sum:sum+n_new_sol[i]]
        sum += n_new_sol[i]

        self.update_archive(candidate_sol=new_sol, candidate_bhv=new_bhv, acq_flag=True)
        self.update_archive(candidate_sol=new_sol, candidate_bhv=new_bhv, pred_flag=True)

        self.visualize_archive(archive=self.acq_archive, acq_flag=True)

    self.obj_t1 = self.obj_archive.stats.num_elites
    self.convergence_errors = n_errors
    gc.collect()
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

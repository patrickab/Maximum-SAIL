import os
import gc
import numpy as np
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter

### Custom Scripts ###
from xfoil.generate_airfoils import generate_parsec_coordinates
from xfoil.simulate_airfoils import xfoil
from utils.pprint_nd import pprint

### Global Parameters ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
BATCH_SIZE = config.BATCH_SIZE
SIGMA_EMITTER = config.SIGMA_EMITTER
SOL_DIMENSION = config.SOL_DIMENSION
BHV_DIMENSION = config.BHV_DIMENSION
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE


def eval_xfoil_loop(self, candidate_sol, evaluate_prediction_archive=False, candidate_acq_or_pred=None):
    """
    Ensures that iter_samples <= BATCH_SIZE are evaluated
    
    XFOIL evaluation is performed in Batches of BATCH_SIZE
        Therefore, if n_samples != BATCH_SIZE, 
        samples need to be evaluated in a loop

    Obj Elites are always stored & rendered in their archive
    Subsequently the GP model is updated with new data

    Finally, the acq/pred archives are updated with obj elites & new gp model
    """

    n_errors = 0
    iteration = 0
    n_new_sol = np.array([])
    new_objectives = np.empty((0, 1))
    new_solutions = np.empty((0, SOL_DIMENSION))
    new_behaviors = np.empty((0, BHV_DIMENSION))
    remaining_samples = candidate_sol.shape[0]

    old_obj_df = self.obj_archive.as_pandas()
    old_obj_solutions = old_obj_df.values[:,4:]
    old_obj_behavior = old_obj_df.values[:,1:3]

    old_acq_df = self.acq_archive.as_pandas()
    old_acq_solutions = old_acq_df.values[:,4:]
    old_acq_behavior = old_acq_df.values[:,1:3]

    old_pred_df = self.pred_archive.as_pandas()
    old_pred_solutions = old_pred_df.values[:,4:]
    old_pred_behavior = old_pred_df.values[:,1:3]

    self.obj_t0 = self.obj_archive.stats.num_elites
    print("Obj Archive Size (before):", self.obj_t0)
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
        _ , surface_batch = generate_parsec_coordinates(i_solutions)

        # evaluate samples batch & extract converged solutions
        _, success_indices, converged_obj = xfoil(i_candidates, surface_batch)
        success_indices = success_indices[:i_candidates]
        converged_sol = i_solutions[success_indices]
        converged_bhv = i_solutions[:,1:3][success_indices] # ToDo: generalize calculate behavior

        # used for printing - in future this can be used for visualizing obj improvements in an archive
        if candidate_acq_or_pred is not None:
            i_converged_acq_or_pred = candidate_acq_or_pred[sample_index:sample_index+BATCH_SIZE][success_indices] if success_indices != [] else []
            converged_acq_or_pred = np.vstack(i_converged_acq_or_pred)

        i_errors = i_candidates - len(success_indices)
        n_errors += i_errors

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

        new_obj = np.vstack(converged_obj)
        new_objectives = np.vstack((new_objectives, new_obj))

    print("\n\nObjective Evaluation Results and Corresponding Acquisitions/Predictions:")
    target_objectives = np.vstack(converged_acq_or_pred) if candidate_acq_or_pred is not None else None
    true_objectives = np.vstack(new_objectives)
    pprint(target_objectives, true_objectives) if candidate_acq_or_pred is not None else pprint(true_objectives)

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
    print("Obj Archive Size (after):", self.obj_t1)
    self.convergence_errors = n_errors
    gc.collect()
    return


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
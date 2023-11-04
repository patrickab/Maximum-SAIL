import os
import gc
import numpy as np

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


def eval_xfoil_loop(self, solution_batch, measures_batch, evaluate_prediction_archive=False, acq_flag=False, pred_flag=False, candidate_targetvalues=None):
    """
    Ensures that iter_samples <= BATCH_SIZE are evaluated
    
    XFOIL evaluation is performed in Batches of BATCH_SIZE
        Therefore, if n_samples != BATCH_SIZE, 
        samples need to be evaluated in a loop

    - Obj Elites are always stored & rendered in their archive
    - Subsequently the GP model is updated with new data
    - In case of acq/pred, the acq/pred archives are 
        1. cleared 
        2. updated with obj elites 
        3. updated under new gp model
            -> this is can be done in order to preserve elites, 
               that remain performant even under a new gp model
    """

    if acq_flag and pred_flag: 
        raise ValueError(f'eval_xfoil_loop: acq_flag AND pred_flag')

    n_errors = 0
    iteration = 0
    new_objectives = np.empty((0, 1))

    n_new_obj_elites = 0 # counter for newly discovered objective elites

    remaining_samples = solution_batch.shape[0]

    old_obj_df = self.obj_archive.as_pandas()
    old_obj_solutions = old_obj_df.values[:,4:]
    old_obj_behavior = old_obj_df.values[:,1:3]

    if acq_flag:
        old_acq_df = self.acq_archive.as_pandas()
        old_acq_solutions = old_acq_df.values[:,4:]
        old_acq_behavior = old_acq_df.values[:,1:3]
        target = "Acquisition"
    if pred_flag:
        old_pred_df = self.pred_archive.as_pandas()
        old_pred_solutions = old_pred_df.values[:,4:]
        old_pred_behavior = old_pred_df.values[:,1:3]
        target = "Prediction"
    if evaluate_prediction_archive:
        target = "Prediction"

    converged_acq_or_pred = np.empty((0, 1))

    obj_t0 = self.obj_archive.stats.num_elites
    while remaining_samples>0: # allows indices [0:10], [10:20], [20:22]

        sample_index = iteration*BATCH_SIZE
        i_solutions = solution_batch[sample_index:sample_index+BATCH_SIZE] # alows indices eg [20:22] to be sampled, if 2 samples are left
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
        converged_bhv = measures_batch[sample_index:sample_index+BATCH_SIZE][success_indices] # ToDo: generalize calculate behavior

        i_errors = i_candidates - len(success_indices)
        n_errors += i_errors

        # Insert -1000 for non_converged samples
        objective_values = np.full(i_candidates, -1, dtype=float)
        objective_values[success_indices] = converged_obj

        new_objectives = np.vstack((new_objectives, np.vstack(objective_values))) if new_objectives.shape[0] != 0 else np.vstack(objective_values)

        if i_errors < 0:
            raise ValueError(f'eval_xfoil_loop: i_errors < 0')
        
        # store new data
        if not evaluate_prediction_archive:

            if converged_sol.shape[0] != np.unique(converged_sol, axis=0).shape[0]:
                raise ValueError(f'eval_xfoil_loop: converged_sol contains duplicates')

            self.update_gp_data(new_solutions=converged_sol, new_objectives=converged_obj)
            self.update_archive(candidate_sol=converged_sol, candidate_obj=converged_obj, candidate_bhv=converged_bhv, obj_flag=True)

            self.visualize_archive(self.new_archive, new_flag=True)
            self.visualize_archive(self.obj_archive, obj_flag=True)

            n_new_obj_elites += self.n_new_obj_elites # if eval_xfoil_loop is called with more than BATCH_SIZE samples, increment iteratively

        else:
            self.update_archive(candidate_sol=converged_sol, candidate_obj=converged_obj, candidate_bhv=converged_bhv, evaluate_prediction_archive=True)


    # within initialization, no "target_values" are available
    if candidate_targetvalues is not None:
        if candidate_targetvalues.shape[0] != 0:
            print(f"\n\nObjective Evaluation Results and Corresponding {target} Values:")

            # ToDo: check target value shape
            target_objectives = np.vstack(np.hstack(candidate_targetvalues))
            true_objectives = np.vstack(new_objectives)
            pprint(target_objectives, true_objectives)
        else:
            print("\n\nNo Converged Solutions")


    if (not evaluate_prediction_archive) and (not self.random_flag):
        self.update_gp_model()


    # update acquisition elites under new gp model
    if acq_flag and not self.random_flag:

        self.acq_archive.clear()

        if self.acq_mes_flag:

            # determine all acquisition elites, that have not been evaluated
            unevaluated_solution_mask = ~np.isin(old_acq_solutions, self.sol_array).all(1)

            # select only unevaluated acquisition elites
            old_sol = old_acq_solutions[unevaluated_solution_mask]
            old_bhv = old_acq_behavior[unevaluated_solution_mask]

        if self.acq_ucb_flag: 
            # preserve all acquisition elites, & add all objective elites
            old_sol = np.concatenate((old_obj_solutions, old_acq_solutions), axis=0)
            old_bhv = np.concatenate((old_obj_behavior, old_acq_behavior), axis=0)

        # update unevaluated elites under new GP
        self.update_archive(candidate_sol=old_sol, candidate_bhv=old_bhv, acq_flag=True)
    

    # update prediction elites under new gp model
    if pred_flag:

        self.pred_archive.clear() if acq_flag else self.pred_archive.clear()
        old_target_solutions = old_pred_solutions
        old_taget_behavior = old_pred_behavior

        old_sol = np.concatenate((old_obj_solutions, old_target_solutions), axis=0)
        old_bhv = np.concatenate((old_obj_behavior, old_taget_behavior), axis=0)

        # update prediction elites under new GP
        self.update_archive(candidate_sol=old_sol, candidate_bhv=old_bhv, pred_flag=True)

    obj_t1 = self.obj_archive.stats.num_elites
    self.convergence_errors = n_errors
    gc.collect()
    return obj_t0, obj_t1, n_new_obj_elites
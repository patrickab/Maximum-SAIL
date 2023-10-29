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
    n_new_sol = np.array([])
    new_objectives = np.empty((0, 1))
    new_solutions = np.empty((0, SOL_DIMENSION))
    new_behaviors = np.empty((0, BHV_DIMENSION))

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

        # used for printing - in future this can be used for visualizing obj improvements in an archive
        # if candidate_targetvalues is not None:
        #     i_converged_acq_or_pred = candidate_targetvalues[sample_index:sample_index+BATCH_SIZE][success_indices] if success_indices != [] else []
        #     converged_acq_or_pred = np.vstack((converged_acq_or_pred, np.vstack(i_converged_acq_or_pred)))

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

        if i_new_sol == 0:
            continue
        
        new_sol = converged_sol
        new_solutions = np.vstack((new_solutions, new_sol))

        new_bhv = converged_bhv
        new_behaviors = np.vstack((new_behaviors, new_bhv))

        new_obj = np.vstack(converged_obj)
        new_objectives = np.vstack((new_objectives, new_obj))

    if evaluate_prediction_archive:
        return

    # within initialization, no "target_values" are available
    if candidate_targetvalues is not None:
        if candidate_targetvalues.shape[0] != 0:
            print("\n\nObjective Evaluation Results and Corresponding Acquisitions/Predictions:")
            target_objectives = np.vstack(candidate_targetvalues)
            true_objectives = np.vstack(new_objectives)
            pprint(target_objectives, true_objectives)
        else:
            print("\n\nNo Converged Solutions")

    # update GP model with new data
    self.update_gp_model()

    if acq_flag or pred_flag: # update acq/pred archives under new gp model
            
        self.acq_archive.clear() if acq_flag else self.pred_archive.clear()
        old_target_solutions = old_acq_solutions if acq_flag else old_pred_solutions  
        old_taget_behavior = old_acq_behavior if acq_flag else old_pred_behavior  

        old_sol = np.concatenate((old_obj_solutions, old_target_solutions), axis=0)
        old_bhv = np.concatenate((old_obj_behavior, old_taget_behavior), axis=0)
        self.update_archive(candidate_sol=old_sol, candidate_bhv=old_bhv, acq_flag=acq_flag, pred_flag=pred_flag)

    sum = 0                   # update acq/pred archives with new obj elites ITERATIVELY to ensure that all videos are of equal length
    for i in range(n_new_sol.shape[0]):
        new_sol = new_solutions[sum:sum+n_new_sol[i]]
        new_bhv = new_behaviors[sum:sum+n_new_sol[i]]
        sum += n_new_sol[i]

        self.update_archive(candidate_sol=new_sol, candidate_bhv=new_bhv, acq_flag=True)
        self.update_archive(candidate_sol=new_sol, candidate_bhv=new_bhv, pred_flag=True) if pred_flag else None

        self.visualize_archive(archive=self.acq_archive, acq_flag=True)
        self.visualize_archive(archive=self.pred_archive, pred_flag=True) # renders empty archive, until prediction verification is reached

    obj_t1 = self.obj_archive.stats.num_elites
    self.convergence_errors = n_errors
    gc.collect()
    return obj_t0, obj_t1
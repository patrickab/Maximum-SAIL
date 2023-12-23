"""
Ensures that iter_samples <= BATCH_SIZE are evaluated

XFOIL evaluation is performed in Batches of BATCH_SIZE
Therefore, if n_samples != BATCH_SIZE, 
samples need to be evaluated in a loop

In order to preserve high performing, non evaluated solutions,
acquisition/prediction archives are updated under new GP model.

    - Allows indices eg [20:22] to be sampled, if 2 samples are left
    - Obj Elites are always stored & rendered in their archive
    - Subsequently the GP model is updated with new data
"""

import os
import gc
import numpy as np

### Custom Scripts ###
from xfoil.generate_airfoils import generate_parsec_coordinates
from acq_functions.acq_mes import simple_mes, acq_mes
from gp.optimize_mean import maximize_mean
from xfoil.simulate_airfoils import xfoil
from utils.pprint_nd import pprint
from sail_runner import SailRun

### Global Parameters ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
BATCH_SIZE = config.BATCH_SIZE
SIGMA_EMITTER = config.SIGMA_EMITTER
SOL_DIMENSION = config.SOL_DIMENSION
BHV_DIMENSION = config.BHV_DIMENSION
OBJ_DIMENSION = config.OBJ_DIMENSION
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
ACQ_MES_MIN_THRESHHOLD = config.ACQ_MES_MIN_THRESHHOLD

def eval_xfoil_loop(self: SailRun, solution_batch, measures_batch, evaluate_prediction_archive=False, acq_flag=False, pred_flag=False, visualize_flag=True, candidate_targetvalues=None):

    target = "Acquisition" if acq_flag else "Prediction"
    new_objectives = np.empty((0, 1))
    obj_t0 = self.obj_archive.stats.num_elites

    n_errors = 0
    iteration = 0
    n_new_obj_elites = 0

    if self.custom_flag and self.obj_current_iteration % 5 == 0 and not evaluate_prediction_archive:
        """Evaluate global max mean prediction"""

        new_x, max_mean = maximize_mean(self.gp_model)
        generate_parsec_coordinates(new_x)
        _, success_index, converged_obj = xfoil(iterations=1)

        if converged_obj.shape[0] == 1:
            self.obj_archive.add_single(new_x[0], converged_obj, measures=new_x[0,1:3])
            print(f"Max Mean: {max_mean} - Max Mean Objective: {converged_obj}")
            self.update_gp_data(new_solutions=new_x, new_objectives=converged_obj)

    if np.any(np.isin(solution_batch, self.sol_array).all(1)):
        raise ValueError("Duplicate Solution Error: Solution Candidate already exist in GP Data")

    for i in range(0, solution_batch.shape[0], BATCH_SIZE):

        i_solutions = solution_batch[i:i+BATCH_SIZE]
        i_solutions = np.vstack(i_solutions)
        n_solutions = i_solutions.shape[0]

        # evaluate samples & extract converged solutions
        _, _ = generate_parsec_coordinates(i_solutions)
        _, success_indices, converged_obj = xfoil(iterations=n_solutions)

        if len(success_indices) == 0:
             continue

        success_indices = success_indices[:n_solutions]
        converged_sol = i_solutions[success_indices]
        converged_bhv = measures_batch[i:i+BATCH_SIZE][success_indices]

        i_errors = n_solutions - len(success_indices)
        n_errors += i_errors

        # Insert -1000 for non_converged samples - used for printing results only
        objective_values = np.full(n_solutions, -1, dtype=float)
        objective_values[success_indices] = converged_obj
        new_objectives = np.vstack((new_objectives, np.vstack(objective_values))) if new_objectives.shape[0] != 0 else np.vstack(objective_values)

        iteration += 1
        if i_errors < 0:
            raise ValueError(f'eval_xfoil_loop: i_errors < 0')

        # store new data
        if not evaluate_prediction_archive:

            self.update_gp_data(new_solutions=converged_sol, new_objectives=converged_obj)
            self.update_archive(candidate_sol=converged_sol, candidate_obj=converged_obj, candidate_bhv=converged_bhv, obj_flag=True)

            if visualize_flag:
                self.visualize_archive(self.new_archive, new_flag=True)
                self.visualize_archive(self.obj_archive, obj_flag=True)

            n_new_obj_elites += self.n_new_obj_elites

        else:
            self.update_archive(candidate_sol=converged_sol, candidate_obj=converged_obj, candidate_bhv=converged_bhv, evaluate_prediction_archive=True)

    if (not evaluate_prediction_archive) and (not self.random_flag):
        self.update_gp_model()

    if np.any(new_objectives >= 5.2):
        # write to csv vile
        print("\nWriting to csv file")
        self.obj_archive.as_pandas(include_solutions=True).to_csv(f"exceptional_obj_archive_{self.domain}_{self.initial_seed}_{self.current_seed}.csv")
        self.acq_archive.as_pandas(include_solutions=True).to_csv(f"exceptional_acq_archive_{self.domain}_{self.initial_seed}_{self.current_seed}.csv")

    if candidate_targetvalues is not None:
        if candidate_targetvalues.shape[0] != 0:

            print(f"\n\nObjective Evaluation Results and Corresponding {target} Values:")
            target_objectives = np.vstack(np.hstack(candidate_targetvalues))
            true_objectives = np.vstack(new_objectives)
            pprint(target_objectives, true_objectives)

    obj_t1 = self.obj_archive.stats.num_elites
    self.convergence_errors = n_errors
    return obj_t0, obj_t1, n_new_obj_elites

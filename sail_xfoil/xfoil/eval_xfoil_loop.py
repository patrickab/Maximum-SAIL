"""
Defines the evaluation routine for XFOIL within SAIL.
Ensures that not more than `BATCH_SIZE` samples are evaluated at once.

1. Express PARSEC encoded `solution_batch`
   as XFOIL compatible coordinate file
2. Evaluate coordinates using XFOIL
3. Store converged solutions in
   a) GP model
   b) Obj Archive
4. Visualize updated archives
5. Update GP model
6. Print results & candidate_targetvalues (if available)

"""

import os
import gc
import numpy as np

### Custom Scripts ###
from xfoil.express_airfoils import generate_parsec_coordinates
from xfoil.simulate_airfoils import xfoil
from utils.pprint_nd import pprint
from sail_run import SailRun


### Global Parameters ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
BATCH_SIZE = config.BATCH_SIZE
SIGMA_EMITTER = config.SIGMA_EMITTER
SOL_DIMENSION = config.SOL_DIMENSION
BHV_DIMENSION = config.BHV_DIMENSION
OBJ_DIMENSION = config.OBJ_DIMENSION
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE

def eval_xfoil_loop(self: SailRun, solution_batch, measures_batch, evaluate_prediction_archive=False, acq_flag=False, pred_flag=False, visualize_flag=True, candidate_targetvalues=None):

    n_errors = 0
    n_new_obj_elites = 0
    new_objectives = np.empty((0, 1))
    obj_t0 = self.obj_archive.stats.num_elites
    target = "Acquisition" if acq_flag else "Prediction"

    for i in range(0, solution_batch.shape[0], BATCH_SIZE):

        i_solutions = solution_batch[i:i+BATCH_SIZE]
        i_solutions = np.vstack(i_solutions)
        n_solutions = i_solutions.shape[0]

        # evaluate samples & extract converged solutions
        generate_parsec_coordinates(i_solutions)
        _, success_indices, converged_obj = xfoil(iterations=n_solutions)

        if len(success_indices) == 0:
            continue

        success_indices = success_indices[:n_solutions]
        converged_sol = i_solutions[success_indices]
        converged_bhv = measures_batch[i:i+BATCH_SIZE][success_indices]

        i_errors = n_solutions - len(success_indices)
        n_errors += i_errors

        # Insert -1000 for non_converged samples - used for printing results
        objective_values = np.full(n_solutions, -1, dtype=float)
        objective_values[success_indices] = converged_obj
        new_objectives = np.vstack((new_objectives, np.vstack(objective_values))) if new_objectives.shape[0] != 0 else np.vstack(objective_values)

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
        if candidate_targetvalues.shape[0] != 0 and new_objectives.shape[0] != 0:

            print(f"\n\nObjective Evaluation Results and Corresponding {target} Values:")
            target_objectives = np.vstack(np.hstack(candidate_targetvalues))
            true_objectives = np.vstack(new_objectives)
            pprint(target_objectives, true_objectives)

    obj_t1 = self.obj_archive.stats.num_elites
    self.convergence_errors = n_errors
    return obj_t0, obj_t1, n_new_obj_elites
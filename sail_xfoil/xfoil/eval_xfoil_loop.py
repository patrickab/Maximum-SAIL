"""
Ensures that iter_samples <= BATCH_SIZE are evaluated

XFOIL evaluation is performed in Batches of BATCH_SIZE
Therefore, if n_samples != BATCH_SIZE, 
samples need to be evaluated in a loop

In order to preserve high performing, non evaluated solutions,
acquisition/prediction archives are updated under new GP model.

Updating routine differs between UCB and MES

    - Allows indices eg [20:22] to be sampled, if 2 samples are left
    - Obj Elites are always stored & rendered in their archive
    - Subsequently the GP model is updated with new data

    In case of UCB or Predictions 
        1. archives gets cleared 
        2. obj elites are added
        3. previous elites are updated under new gp model

    In case of MES
        1. archive gets cleared
        2. 40 best obj elites are added  -  (setting obj value to ACQ_MES_MIN_THRESHHOLD*1.001)
        3. previous elites are updated under new gp model
"""

import os
import gc
import numpy as np

### Custom Scripts ###
from xfoil.generate_airfoils import generate_parsec_coordinates
from acq_functions.acq_mes import simple_mes, acq_mes
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

def eval_xfoil_loop(self: SailRun, solution_batch, measures_batch, evaluate_prediction_archive=False, acq_flag=False, pred_flag=False, candidate_targetvalues=None):

    n_errors = 0
    iteration = 0
    n_new_obj_elites = 0
    new_objectives = np.empty((0, 1))
    remaining_samples = solution_batch.shape[0]
    obj_t0 = self.obj_archive.stats.num_elites
    target = "Acquisition" if acq_flag else "Prediction"

    while remaining_samples>0:

        sample_index = iteration*BATCH_SIZE
        iter_solutions = solution_batch[sample_index:sample_index+BATCH_SIZE]
        iter_solutions = np.vstack(iter_solutions)
        n_solutions = iter_solutions.shape[0]

        # evaluate samples & extract converged solutions
        _, surface_batch = generate_parsec_coordinates(iter_solutions)
        _, success_indices, converged_obj = xfoil(iterations=n_solutions, surface_batch=surface_batch)
        success_indices = success_indices[:n_solutions]
        converged_sol = iter_solutions[success_indices]
        converged_bhv = measures_batch[sample_index:sample_index+BATCH_SIZE][success_indices]

        remaining_samples -= n_solutions
        i_errors = n_solutions - len(success_indices)
        n_errors += i_errors

        # Insert -1000 for non_converged samples - used for printing results
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

            self.visualize_archive(self.new_archive, new_flag=True)
            self.visualize_archive(self.obj_archive, obj_flag=True)

            n_new_obj_elites += self.n_new_obj_elites

        else:
            self.update_archive(candidate_sol=converged_sol, candidate_obj=converged_obj, candidate_bhv=converged_bhv, evaluate_prediction_archive=True)

    if np.any(new_objectives >= 5.2):
        # write to csv vile
        print("\nWriting to csv file")
        self.obj_archive.as_pandas(include_solutions=True).to_csv(f"exceptional_obj_archive_{self.domain}_{self.initial_seed}_{self.current_seed}.csv")

    if candidate_targetvalues is not None:
        if candidate_targetvalues.shape[0] != 0:

            print(f"\n\nObjective Evaluation Results and Corresponding {target} Values:")
            target_objectives = np.vstack(np.hstack(candidate_targetvalues))
            true_objectives = np.vstack(new_objectives)
            pprint(target_objectives, true_objectives)


    if (not evaluate_prediction_archive) and (not self.random_flag):
        if solution_batch.shape[0] >= 800:
            self.obj_archive.as_pandas(include_solutions=True).to_csv(f"sobol_10000.csv")
            self.sol_array = np.empty((0, SOL_DIMENSION))
            self.obj_array = np.empty((0, OBJ_DIMENSION))
            sol = self.obj_archive.as_pandas(include_solutions=True).solution_batch()
            obj = self.obj_archive.as_pandas(include_solutions=True).objective_batch()
            self.update_gp_data(new_solutions=sol, new_objectives=obj)
        self.update_gp_model()


    if acq_flag:

        if self.vanilla_flag:

            obj_elite_df = self.obj_archive.as_pandas(include_solutions=True).sort_values(by='objective', ascending=False)
            self.acq_archive.clear()
            self.acq_archive.add(obj_elite_df.solution_batch(), obj_elite_df.objective_batch(), obj_elite_df.measures_batch())

        if self.custom_flag:

            print("acq archive size before update: ", self.acq_archive.stats.num_elites)

            obj_elite_df = self.obj_archive.as_pandas(include_solutions=True).sort_values(by='objective', ascending=False)
            acq_elite_df = self.acq_archive.as_pandas(include_solutions=True).sort_values(by='objective', ascending=False)
            
            if self.acq_mes_flag:
                n_bins = np.prod(self.acq_archive.dims)
                acq_elite_df = acq_elite_df.head(n_bins//3)

            acq_elites_solutions = acq_elite_df.solution_batch()
            acq_elites_measures = acq_elite_df.measures_batch()

            obj_elites_solutions = obj_elite_df.solution_batch()
            obj_elites_objectives = obj_elite_df.objective_batch() if self.acq_ucb_flag else np.full(obj_elites_solutions.shape[0], ACQ_MES_MIN_THRESHHOLD*1.001)
            obj_elites_measures = obj_elite_df.measures_batch()

            self.acq_archive.clear()
            self.update_archive(candidate_sol=acq_elites_solutions, candidate_bhv=acq_elites_measures, acq_flag=True)
            print(self.acq_archive.as_pandas(include_solutions=True).sort_values(by='objective', ascending=False).head(20).objective_batch())

            print("acq archive size after update: ", self.acq_archive.stats.num_elites)

            if self.acq_archive.stats.num_elites < 10 or self.acq_ucb_flag:
                if self.acq_mes_flag:
                    obj_elite_df = obj_elite_df.head(int(n_bins*0.05))
                    obj_elite_df = obj_elite_df.sample(n=BATCH_SIZE, random_state=self.current_seed, replace=True)
                self.acq_archive.add(obj_elites_solutions, obj_elites_objectives, obj_elites_measures)

        print("Best Acq Objective: ", self.acq_archive.as_pandas(include_solutions=True).sort_values(by='objective', ascending=False).head(1).objective_batch())
        print("Worst Acq Objective: ", self.acq_archive.as_pandas(include_solutions=True).sort_values(by='objective', ascending=False).tail(1).objective_batch())

    if pred_flag:

        obj_elite_df = self.obj_archive.as_pandas(include_solutions=True).sort_values(by='objective', ascending=False)
        pred_elite_df = self.pred_archive.as_pandas(include_solutions=True).sort_values(by='objective', ascending=False)
        self.pred_archive.clear()

        # Update prediction elites under new GP
        self.pred_archive.add(obj_elite_df.solution_batch(), obj_elite_df.objective_batch(), obj_elite_df.measures_batch())
        self.update_archive(candidate_sol=pred_elite_df.solution_batch(), candidate_bhv=pred_elite_df.measures_batch(), pred_flag=True)

    # Remove all variables from RAM and Cache, that are not needed anymore 
    # list all variables in dir() on console
    del candidate_targetvalues, converged_bhv, converged_obj, converged_sol, i_errors, iteration, n_solutions, remaining_samples, sample_index, success_indices, target, objective_values, new_objectives
    gc.collect()

    obj_t1 = self.obj_archive.stats.num_elites
    self.convergence_errors = n_errors
    return obj_t0, obj_t1, n_new_obj_elites

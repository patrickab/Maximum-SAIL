###### Import Foreign Scripts ######
from gp.predict_objective import predict_objective
from ribs.archives import GridArchive
from ribs.archives import ArchiveDataFrame
import numpy as np
import pandas
import gc
import os

### Configurable Variables ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), 'config', 'config.ini'))
PREDICTION_VERIFICATIONS = config.PREDICTION_VERIFICATIONS
MAX_PRED_VERIFICATION = config.MAX_PRED_VERIFICATION
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
ACQ_N_OBJ_EVALS = config.ACQ_N_OBJ_EVALS
ACQ_N_MAP_EVALS = config.ACQ_N_MAP_EVALS
SOL_DIMENSION = config.SOL_DIMENSION
INIT_N_EVALS = config.INIT_N_EVALS
PRED_N_EVALS = config.PRED_N_EVALS
BATCH_SIZE = config.BATCH_SIZE

n_obj_evals = INIT_N_EVALS + ACQ_N_OBJ_EVALS + MAX_PRED_VERIFICATION
CSV_BUFFERSIZE = (n_obj_evals/BATCH_SIZE) / 8

###### Import Custom Scripts ######

from utils.utils import eval_xfoil_loop
from utils.pprint_nd import pprint

from xfoil.generate_airfoils import generate_parsec_coordinates
from xfoil.simulate_airfoils import xfoil

from acq_functions.acq_ucb import acq_ucb
from map_elites import map_elites

from sail_runner import SailRun



def run_custom_sail(self: SailRun):
    """
    Extra evaluations are given if eval_pred_flag is True (see sail.py)
        if not eval_pred_flag, extra_evals = 0
    """

    CURIOSITY = 6 # For Hybrid Approach: 'CURIOSITY//BATCH_SIZE' new bin elites are to be sampled


    iteration = 1

    mean_acq_improvement = 0 # ToDo
    mean_obj_improvement = 0 # ToD
    consumed_obj_evals = 0
    total_new_obj_bins = 0
    total_new_acq_bins = 0
    total_new_obj_elites = 0
    total_new_acq_elites = 0
    total_obj_improvements = 0
    total_acq_improvements = 0
    total_convergence_errors = 0

    total_eval_budget = ACQ_N_OBJ_EVALS if self.pred_verific_flag else ACQ_N_OBJ_EVALS + MAX_PRED_VERIFICATION # if no budget for prediction verification is given, add MAX_PRED_VERIFICATION to ACQ_N_MAP_EVALS to ensure equal number of evaluations
    total_acq_eval_budget = ACQ_N_MAP_EVALS * (ACQ_N_OBJ_EVALS//BATCH_SIZE)
    current_eval_budget = total_eval_budget
    current_acq_eval_budget = total_acq_eval_budget

    while(current_eval_budget >= BATCH_SIZE):

        if consumed_obj_evals % (total_eval_budget//2) == 0 and consumed_obj_evals != 0:
            print("\nDECREASING CURIOSITY PARAMETER\n")
            CURIOSITY -= 2

        test_acq_t0 = self.acq_archive.stats.num_elites
        new_acq_elites, acq_t0, acq_t1 = map_elites(self, acq_flag=True)                          # Produce new acquisition elites
        test_acq_t1 = self.acq_archive.stats.num_elites

        if acq_t0 != test_acq_t0 or acq_t1 != test_acq_t1:
            raise ValueError("Acq Archive Size Mismatch")

        if new_acq_elites.stats.num_elites < BATCH_SIZE: ensure_n_new_elites(self=self, new_elite_archive=new_acq_elites, acq_flag=True)      # Sample until enough new acquisition elites are found
        improved_elites, new_bin_elites = maximize_improvement(self=self, new_elite_archive=new_acq_elites, old_elite_archive=self.obj_archive)          # Split new elites into improved elites & new bin elites & calculate objective improvement
        candidate_solutions_df = select_samples(self, improved_elites=improved_elites, new_bin_elites=new_bin_elites, acq_flag=True, curiosity=CURIOSITY)          # Select samples based on exploration behavior defined in the class constructor
        solution_batch = candidate_solutions_df.solution_batch()
        objective_batch = candidate_solutions_df.objective_batch()
        measures_batch = candidate_solutions_df.measures_batch()
        obj_t0, obj_t1 = eval_xfoil_loop(self, solution_batch=solution_batch, measures_batch=measures_batch, candidate_targetvalues=objective_batch, acq_flag=True)       # Evaluate Acquisition Elites & Update Acq Archive under resulting GP Model

        current_eval_budget -= BATCH_SIZE
        current_acq_eval_budget -= ACQ_N_MAP_EVALS
        consumed_obj_evals = total_eval_budget - current_eval_budget
        consumed_acq_evals = total_acq_eval_budget - current_acq_eval_budget
        
        # Count newly discovered elites
        n_new_obj_bins = obj_t1 - obj_t0
        n_new_acq_bins = new_bin_elites.shape[0]
        n_new_obj_elites = self.new_obj.shape[0]
        n_new_acq_elites = improved_elites.shape[0] + new_bin_elites.shape[0]
        n_new_obj_improvements = n_new_obj_elites - n_new_obj_bins
        n_new_acq_improvements = n_new_acq_elites - n_new_acq_bins
        # Sum newly discovered elites
        total_new_obj_bins += n_new_obj_bins
        total_new_acq_bins += n_new_acq_bins
        total_new_obj_elites += n_new_obj_elites
        total_new_acq_elites += n_new_acq_elites
        total_obj_improvements += n_new_obj_improvements
        total_acq_improvements += n_new_acq_improvements
        # Calculate percentages
        percentage_new_obj_bins = (total_new_obj_bins/consumed_obj_evals)*100
        percentage_new_acq_bins = (total_new_acq_bins/consumed_acq_evals)*100
        percentage_new_obj_elites = (total_new_obj_elites/consumed_obj_evals)*100
        percentage_new_acq_elites = (total_new_acq_elites/consumed_acq_evals)*100
        percentage_obj_improvements   = (total_obj_improvements/consumed_obj_evals)*100
        percentage_acq_improvements   = (total_acq_improvements/consumed_acq_evals)*100

        convergence_errors = self.convergence_errors
        total_convergence_errors += convergence_errors
        percentage_convergence_errors = (total_convergence_errors/consumed_obj_evals)*100

        qd_obj = sum(self.obj_archive.as_pandas(include_solutions=True)['objective'].values)
        qd_acq = sum(self.acq_archive.as_pandas(include_solutions=True)['objective'].values)
        n_bins = np.prod(self.obj_archive.dims)

        obj_qd_per_bin = round(qd_obj/n_bins, 1)
        acq_qd_per_bin = round(qd_acq/n_bins, 1)
        obj_qd_per_elite = round(qd_obj/self.obj_archive.stats.num_elites, 1)
        acq_qd_per_elite = round(qd_acq/self.acq_archive.stats.num_elites, 1)

        # Print & Store Anytime Metrics
        anytime_metrics(self, acq_flag=True, iteration=iteration, current_eval_budget=current_eval_budget, consumed_obj_evals=consumed_obj_evals, obj_t0=obj_t0, obj_t1=obj_t1, obj_qd_per_elite=obj_qd_per_elite, obj_qd_per_bin=obj_qd_per_bin, n_new_obj_bins=n_new_obj_bins, n_new_obj_improvements=n_new_obj_improvements, n_new_obj_elites=n_new_obj_elites, percentage_convergence_errors=percentage_convergence_errors, percentage_new_obj_bins=percentage_new_obj_bins, percentage_obj_improvements=percentage_obj_improvements, percentage_new_obj_elites=percentage_new_obj_elites, total_new_obj_bins=total_new_obj_bins, total_obj_improvements=total_obj_improvements, total_new_obj_elites=total_new_obj_elites, total_convergence_errors=total_convergence_errors, convergence_errors=convergence_errors, n_new_target_bins=n_new_acq_bins, n_new_target_improvements=n_new_acq_improvements, n_new_target_elites=n_new_acq_elites, percentage_new_target_bins=percentage_new_acq_bins, percentage_target_improvements=percentage_acq_improvements, percentage_new_target_elites=percentage_new_acq_elites, total_new_target_bins=total_new_acq_bins, total_target_improvements=total_acq_improvements, total_new_target_elites=total_new_acq_elites, target_t0=acq_t0, target_t1=acq_t1, target_qd_per_bin=acq_qd_per_bin, target_qd_per_elite=acq_qd_per_elite)

        iteration += 1

        if iteration % 20 == 0:
            gc.collect()

    return


def anytime_metrics(self, iteration, current_eval_budget, consumed_obj_evals, obj_t0, obj_t1, obj_qd_per_elite, obj_qd_per_bin, n_new_obj_bins, n_new_obj_improvements, n_new_obj_elites, percentage_convergence_errors, percentage_new_obj_bins, percentage_obj_improvements, percentage_new_obj_elites, total_new_obj_bins, total_obj_improvements, total_new_obj_elites, total_convergence_errors, convergence_errors, n_new_target_bins, n_new_target_improvements, n_new_target_elites, percentage_new_target_bins, percentage_target_improvements, percentage_new_target_elites, total_new_target_bins, total_target_improvements, total_new_target_elites, target_t0, target_t1, target_qd_per_bin, target_qd_per_elite, acq_flag=False, pred_flag=False):

    if acq_flag:
        target = "Acq "
    if pred_flag:
        target = "Pred"

    anytime_dtypes = {'Remaining Obj Evals': int, 'Consumed Obj Evals': int, 'Obj Archive Size (before)': int, 'Obj Archive Size (after)': int, 'Obj QD (per bin)': float, 'Obj QD (per elite)': float, 'New Convergence Errors': int, 'New Obj Bins': int, 'New Obj Improvements': int, 'New Obj Elites': int, 'Percentage Convergence Errors': float, 'Percentage New Obj Bins': float, 'Percentage Obj Improvements': float, 'Percentage New Obj Elites': float, 'Total Convergence Errors': int, 'Total New Obj Bins': int, 'Total Obj Improvements': int, 'Total New Obj Elites': int, f'{target} Archive Size (before)': int, f'{target} Archive Size (after)': int, f'{target} QD (per bin)': float, f'{target} QD (per elite)': float, 'Newf {target} Bins': int, 'Newf {target} Improvements': int, 'Newf {target} Elites': int, 'Percentage Newf {target} Bins': float, 'Percentagef {target} Improvements': float, 'Percentage Newf {target} Elites': float, 'Total Newf {target} Bins': int, 'Totalf {target} Improvements': int, 'Total Newf {target} Elites': int}
    anytime_metrics = pandas.DataFrame({c: pandas.Series(dtype=d) for c, d in anytime_dtypes.items()})
    pandas.options.display.float_format = '{:.2f}'.format

    print(f"\n\nRemaining Obj Evals           : {current_eval_budget}")
    print(f"Consumed Obj Evals            : {consumed_obj_evals}\n")

    print(f"Obj Archive Size (before)     : {obj_t0}")
    print(f"Obj Archive Size  (after)     : {obj_t1}")
    print(f"Obj QD (per elite)            : {obj_qd_per_elite}")
    print(f"Obj QD (per bin)              : {obj_qd_per_bin}\n")

    print(f"Iteration New Obj Bins        : {n_new_obj_bins}")
    print(f"Iteration Improvements        : {n_new_obj_improvements}")
    print(f"Iteration New Obj Elites      : {n_new_obj_elites}")
    print(f"Percentage Convergence Errors : {percentage_convergence_errors:.1f}%")
    print(f"Percentage New Obj Bins       : {percentage_new_obj_bins:.1f}%")
    print(f"Percentage Obj Improvements   : {percentage_obj_improvements:.1f}%")
    print(f"Percentage New Obj Elites     : {percentage_new_obj_elites:.1f}%")
    print(f"Total New Obj Bins            : {total_new_obj_bins}")
    print(f"Total Improvements            : {total_obj_improvements}")
    print(f"Total New Obj Elites          : {total_new_obj_elites}\n")

    print(f"Total Convergence Errors      : {total_convergence_errors}")
    print(f"Iteration Convergence Errors  : {convergence_errors}\n")

    print(f"Iteration New {target} Bins       : {n_new_target_bins}")
    print(f"Iteration Improvements        : {n_new_target_improvements}")
    print(f"Iteration New {target} Elites     : {n_new_target_elites}")
    print(f"Percentage New {target} Bins      : {percentage_new_target_bins:.1f}%")
    print(f"Percentage {target} Improvements  : {percentage_target_improvements:.1f}%")
    print(f"Percentage New {target} Elites    : {percentage_new_target_elites:.1f}%")
    print(f"Total New {target} Bins           : {total_new_target_bins}")
    print(f"Total Improvements            : {total_target_improvements}")
    print(f"Total New {target} Elites         : {total_new_target_elites}\n")

    print(f"{target} Archive Size (before)   : {target_t0}")
    print(f"{target} Archive Size  (after)   : {target_t1}")
    print(f"New {target} Elites              : {n_new_target_elites}\n")

    print(f"{target} QD (per bin)            : {target_qd_per_bin}")
    print(f"New {target} Bins                : {n_new_target_bins}")
    print(f"Mean {target} QD                 : {target_qd_per_elite}\n")

    anytime_data = [current_eval_budget, consumed_obj_evals, obj_t0, obj_t1, obj_qd_per_bin, obj_qd_per_elite, convergence_errors, n_new_obj_bins, n_new_obj_improvements, n_new_obj_elites, percentage_convergence_errors, percentage_new_obj_bins, percentage_obj_improvements, percentage_new_obj_elites, total_convergence_errors, total_new_obj_bins, total_obj_improvements, total_new_obj_elites, target_t0, target_t1, target_qd_per_bin, target_qd_per_elite, n_new_target_bins, n_new_target_improvements, n_new_target_elites, percentage_new_target_bins, percentage_target_improvements, percentage_new_target_elites, total_new_target_bins, total_target_improvements, total_new_target_elites]
    anytime_metrics.loc[iteration] = anytime_data

    if iteration % 2 == 0:
        # Save Anytime Metrics to CSV. If no csv is saved create new csv, else append to existing csv
        target = "acq" if acq_flag else "pred"
        try:
            anytime_metrics.to_csv(f'{self.initial_seed}_{self.domain}_{target}_loop_anytime_metrics.csv', mode='a', header=False, index=True)
        except:
            anytime_metrics.to_csv(f'{self.initial_seed}_{self.domain}_{target}_loop_anytime_metrics.csv', header=True, index=True)
        anytime_metrics = pandas.DataFrame({c: pandas.Series(dtype=d) for c, d in anytime_dtypes.items()})


def ensure_n_new_elites(self: SailRun, new_elite_archive, acq_flag=False, pred_flag=False):

    """
    - Ensures that the appropiate number of candidate solutions are availible for evaluation
    - After 3 extra MAP-Loops, the function returns the best elites found so far to avoid infinite loops

    Inputs:
        New Elite Archive
        Acq Flag XOR Pred Flag
    """

    target = "Acq" if acq_flag else "Pred"

    if acq_flag:
        n_samples = BATCH_SIZE
    if pred_flag and self.pred_verific_flag:
        n_samples = MAX_PRED_VERIFICATION//PREDICTION_VERIFICATIONS
    if pred_flag and not self.pred_verific_flag:
        raise ValueError("Maximize Improvement: Prediction Flag is True, but Prediction Verification Flag is False")

    # Re-enter MAP-Elites (acq/obj) up to 2 times in order to produce new elites 
    iteration = 0
    while new_elite_archive.stats.num_elites < n_samples and iteration <= 3:
        iteration += 1

        print(f'\n\nNot enough {target} Improvements: Re-entering {target}')
        print(f'New {target} Elites (before): {new_elite_archive.stats.num_elites}')
        new_elite_archive, _, _ = map_elites(self, new_elite_archive=new_elite_archive, acq_flag=acq_flag, pred_flag=pred_flag)
        print(f'New {target} Elites (after):  {new_elite_archive.stats.num_elites}')

    return new_elite_archive


def maximize_improvement(self: SailRun, new_elite_archive: GridArchive, old_elite_archive: GridArchive, pred_flag=False):
    """
    - extracts all elites from new_elite_archive
    - splits them into improved elites and new bin elites
    - orders them by objective improvement

    Inputs:
        Old Elites Archive
        New Elite Archive
    """

    old_elite_df = old_elite_archive.as_pandas(include_solutions=True).sort_values(by=['index'])
    new_elite_df = new_elite_archive.as_pandas(include_solutions=True).sort_values(by=['index'])

    is_improved_new_elite = np.isin(new_elite_df['index'], old_elite_df['index'])
    
    improved_elites = new_elite_df[is_improved_new_elite]
    new_bin_elites   = new_elite_df[~is_improved_new_elite]

    # Select old elites that have been improved
    is_improved_old_elite = np.isin(old_elite_df['index'], improved_elites['index'])

    # Calculate objective improvement
    improved_old_elites = old_elite_df[is_improved_old_elite]

    if self.acq_mes_flag and not pred_flag:
        improved_elites = improved_elites.assign(objective_improvement = np.array(improved_elites['objective'])).sort_values(by=['objective_improvement'], ascending=False)
    if self.acq_ucb_flag or pred_flag:
        improved_elites = improved_elites.assign(objective_improvement = np.array(improved_elites['objective'] - np.array(improved_old_elites['objective']))).sort_values(by=['objective_improvement'], ascending=False)
    
    new_bin_elites = new_bin_elites.assign(objective_improvement = np.array(new_bin_elites['objective']))

    return improved_elites, new_bin_elites


def select_samples(self: SailRun, improved_elites, new_bin_elites, acq_flag=False, pred_flag=False, curiosity=5):
    """Selects samples based on exploration behavior defined in the class constructor"""

    if acq_flag:
        target = "Acquisition Values"
        n_samples = BATCH_SIZE
    if pred_flag:
        target = "Prediction Values"
        n_samples = MAX_PRED_VERIFICATION//PREDICTION_VERIFICATIONS

    if self.explore_flag: # Evaluate new_elites first
        candidate_elite_df = pandas.concat([new_bin_elites, improved_elites]).head(n_samples)

    if self.greedy_flag: # Evaluate only maximum improvement, regardeless of new/old bin
        candidate_elite_df = pandas.concat([new_bin_elites, improved_elites]).sort_values(by=['objective_improvement'], ascending=False).head(n_samples)

    if self.hybrid_flag: # Evenly balance sampling of best new_bin_elites & best improved_elites
        n_new_bin_elites = new_bin_elites.shape[0]
        n_improved_elites = improved_elites.shape[0]

        n_new_bin_samples = round((curiosity/10)*n_samples)
        n_improved_samples = n_samples - n_new_bin_samples

        new_bin_elites = new_bin_elites.sort_values(by=['objective_improvement'], ascending=False)
        improved_elites = improved_elites.sort_values(by=['objective_improvement'], ascending=False)

        if n_new_bin_elites >= n_new_bin_samples and n_improved_elites >= n_improved_samples:
            new_bin_elites = new_bin_elites.sample(n=n_new_bin_samples, random_state=self.initial_seed)
            candidate_elite_df = pandas.concat([new_bin_elites.head(n_new_bin_samples), improved_elites.head(n_improved_samples)])
        else:
            if n_new_bin_elites < n_new_bin_samples:
                new_bin_elites = new_bin_elites.sample(n=n_new_bin_elites, random_state=self.initial_seed)
                candidate_elite_df = pandas.concat([new_bin_elites, improved_elites.head(n_samples - n_new_bin_elites)])
            else:
                candidate_elite_df = pandas.concat([new_bin_elites.head(n_samples - n_improved_elites), improved_elites])

    print(f"\n\n: {target} for upcoming Objective Evaluations:\n")
    target_objective = candidate_elite_df['objective'].to_numpy()
    target_objective_improvement = candidate_elite_df['objective_improvement'].to_numpy()
    pprint(target_objective, target_objective_improvement)

    return candidate_elite_df


def prediction_verification_loop(self: SailRun):
    """
    During Prediction, stop after a specified number of evaluations and verify predictions
    """

    print("\n\n ## Enter Prediction Verification Loop##")

    iteration = 1

    mean_pred_improvement = 0 # ToDo
    mean_obj_improvement = 0 # ToDo
    total_new_obj_bins = 0
    total_new_pred_bins = 0
    total_new_obj_elites = 0
    total_new_pred_elites = 0
    total_obj_improvements = 0
    total_pred_improvements = 0
    total_convergence_errors = 0

    total_eval_budget = MAX_PRED_VERIFICATION
    total_pred_eval_budget = PRED_N_EVALS
    current_eval_budget = total_eval_budget
    current_pred_eval_budget = total_pred_eval_budget
    iter_evals = MAX_PRED_VERIFICATION//(PREDICTION_VERIFICATIONS)

    while(current_eval_budget >= iter_evals):

        new_pred_elites, pred_t0, pred_t1 = map_elites(self, pred_flag=True)                          # Produce new preduisition elites

        if new_pred_elites.stats.num_elites < BATCH_SIZE: ensure_n_new_elites(self=self, new_elite_archive=new_pred_elites, pred_flag=True)           # Sample until enough new preduisition elites are found

        improved_elites, new_bin_elites = maximize_improvement(self=self, new_elite_archive=new_pred_elites, old_elite_archive=self.obj_archive, pred_flag=True)      # Split new elites into improved elites & new bin elites & calculate objective improvement
        candidate_solutions_df = select_samples(self, improved_elites=improved_elites, new_bin_elites=new_bin_elites, pred_flag=True, curiosity=4)         # Select samples based on exploration behavior defined in the class constructor
        solution_batch = candidate_solutions_df.solution_batch()
        objective_batch = candidate_solutions_df.objective_batch()
        measures_batch = candidate_solutions_df.measures_batch()
        obj_t0, obj_t1 = eval_xfoil_loop(self, solution_batch=solution_batch, measures_batch=measures_batch, candidate_targetvalues=objective_batch, pred_flag=True)     # Evaluate Acquisition Elites & Update Acq Archive under resulting GP Model

        self.visualize_archive(self.pred_archive, pred_flag=True)

        current_eval_budget -= iter_evals
        current_pred_eval_budget -= PRED_N_EVALS//(MAX_PRED_VERIFICATION+1) # +1 because after the last prediction verification we predict once more

        consumed_obj_evals = total_eval_budget - current_eval_budget
        consumed_pred_evals = total_pred_eval_budget - current_pred_eval_budget
        
        # Count newly discovered elites
        n_new_obj_bins = obj_t1 - obj_t0
        n_new_pred_bins = new_bin_elites.shape[0]
        n_new_obj_elites = self.new_obj.shape[0]
        n_new_pred_elites = improved_elites.shape[0] + new_bin_elites.shape[0]
        n_new_obj_improvements = n_new_obj_elites - n_new_obj_bins
        n_new_pred_improvements = n_new_pred_elites - n_new_pred_bins
        # Sum newly discovered elites
        total_new_obj_bins += n_new_obj_bins
        total_new_pred_bins += n_new_pred_bins
        total_new_obj_elites += n_new_obj_elites
        total_new_pred_elites += n_new_pred_elites
        total_obj_improvements += n_new_obj_improvements
        total_pred_improvements += n_new_pred_improvements
        # Calculate percentages
        percentage_new_obj_bins = (total_new_obj_bins/consumed_obj_evals)*100
        percentage_new_pred_bins = (total_new_pred_bins/consumed_pred_evals)*100
        percentage_new_obj_elites = (total_new_obj_elites/consumed_obj_evals)*100
        percentage_new_pred_elites = (total_new_pred_elites/consumed_pred_evals)*100
        percentage_obj_improvements = (total_obj_improvements/consumed_obj_evals)*100
        percentage_pred_improvements = (total_pred_improvements/consumed_pred_evals)*100

        convergence_errors = self.convergence_errors
        total_convergence_errors += convergence_errors
        percentage_convergence_errors = (total_convergence_errors/consumed_obj_evals)*100

        qd_obj = sum(self.obj_archive.as_pandas(include_solutions=True)['objective'].values)
        qd_pred = sum(self.pred_archive.as_pandas(include_solutions=True)['objective'].values)
        n_bins = np.prod(self.obj_archive.dims)

        obj_qd_per_bin = round(qd_obj/n_bins, 1)
        pred_qd_per_bin = round(qd_pred/n_bins, 1)
        obj_qd_per_elite = round(qd_obj/self.obj_archive.stats.num_elites, 1)
        pred_qd_per_elite = round(qd_pred/self.pred_archive.stats.num_elites, 1)

        # Print & Store Anytime Metrics
        anytime_metrics(self, pred_flag=True, iteration=iteration, current_eval_budget=current_eval_budget, consumed_obj_evals=consumed_obj_evals, obj_t0=obj_t0, obj_t1=obj_t1, obj_qd_per_elite=obj_qd_per_elite, obj_qd_per_bin=obj_qd_per_bin, n_new_obj_bins=n_new_obj_bins, n_new_obj_improvements=n_new_obj_improvements, n_new_obj_elites=n_new_obj_elites, percentage_convergence_errors=percentage_convergence_errors, percentage_new_obj_bins=percentage_new_obj_bins, percentage_obj_improvements=percentage_obj_improvements, percentage_new_obj_elites=percentage_new_obj_elites, total_new_obj_bins=total_new_obj_bins, total_obj_improvements=total_obj_improvements, total_new_obj_elites=total_new_obj_elites, total_convergence_errors=total_convergence_errors, convergence_errors=convergence_errors, n_new_target_bins=n_new_pred_bins, n_new_target_improvements=n_new_pred_improvements, n_new_target_elites=n_new_pred_elites, percentage_new_target_bins=percentage_new_pred_bins, percentage_target_improvements=percentage_pred_improvements, percentage_new_target_elites=percentage_new_pred_elites, total_new_target_bins=total_new_pred_bins, total_target_improvements=total_pred_improvements, total_new_target_elites=total_new_pred_elites, target_t0=pred_t0, target_t1=pred_t1, target_qd_per_bin=pred_qd_per_bin, target_qd_per_elite=pred_qd_per_elite)
            
        iteration += 1

        if iteration % 20 == 0:
            gc.collect()

    new_pred_elite_archive, pred_t0, pred_t1 = map_elites(self, pred_flag=True)


    return self.pred_archive


def run_vanilla_sail(self: SailRun):

    iteration = 1

    mean_acq_improvement = 0 # ToDo
    mean_obj_improvement = 0 # ToDo
    total_new_obj_bins = 0
    total_new_acq_bins = 0
    total_new_obj_elites = 0
    total_new_acq_elites = 0
    total_obj_improvements = 0
    total_acq_improvements = 0
    total_convergence_errors = 0

    total_eval_budget = ACQ_N_OBJ_EVALS + MAX_PRED_VERIFICATION
    total_acq_eval_budget = ACQ_N_MAP_EVALS * (ACQ_N_OBJ_EVALS//BATCH_SIZE)
    current_eval_budget = total_eval_budget
    current_acq_eval_budget = total_acq_eval_budget

    while(current_eval_budget >= BATCH_SIZE):

        new_acq_elites, acq_t0, acq_t1 = map_elites(self, acq_flag=True)                          # Produce new acquisition elites

        if new_acq_elites.stats.num_elites < BATCH_SIZE: ensure_n_new_elites(self=self, new_elite_archive=new_acq_elites, acq_flag=True)      # Sample until enough new acquisition elites are found
        candidate_solutions_df = self.acq_archive.sample_elites(n=BATCH_SIZE)                                                                 # Select samples based on exploration behavior defined in the class constructor
        solution_batch = candidate_solutions_df.solution_batch
        objective_batch = candidate_solutions_df.objective_batch
        measures_batch = candidate_solutions_df.measures_batch
        obj_t0, obj_t1 = eval_xfoil_loop(self, solution_batch=solution_batch, measures_batch=measures_batch, acq_flag=True, candidate_targetvalues=objective_batch)                   # Evaluate Acquisition Elites & Update Acq Archive under resulting GP Model

        current_eval_budget -= BATCH_SIZE
        current_acq_eval_budget -= ACQ_N_MAP_EVALS
        consumed_obj_evals = total_eval_budget - current_eval_budget
        consumed_acq_evals = total_acq_eval_budget - current_acq_eval_budget
        
        # Count newly discovered elites
        n_new_obj_bins = obj_t1 - obj_t0
        n_new_obj_elites = self.new_obj.shape[0]
        n_new_obj_improvements = n_new_obj_elites - n_new_obj_bins
        # Sum newly discovered elites
        total_new_obj_bins += n_new_obj_bins
        total_new_obj_elites += n_new_obj_elites
        total_obj_improvements += n_new_obj_improvements
        # Calculate percentages
        percentage_new_obj_bins = (total_new_obj_bins/consumed_obj_evals)*100
        percentage_new_acq_bins = (total_new_acq_bins/consumed_acq_evals)*100
        percentage_new_obj_elites = (total_new_obj_elites/consumed_obj_evals)*100
        percentage_new_acq_elites = (total_new_acq_elites/consumed_acq_evals)*100
        percentage_obj_improvements   = (total_obj_improvements/consumed_obj_evals)*100
        percentage_acq_improvements   = (total_acq_improvements/consumed_acq_evals)*100

        convergence_errors = self.convergence_errors
        total_convergence_errors += convergence_errors
        percentage_convergence_errors = (total_convergence_errors/consumed_obj_evals)*100

        qd_obj = sum(self.obj_archive.as_pandas(include_solutions=True)['objective'].values)
        qd_acq = sum(self.acq_archive.as_pandas(include_solutions=True)['objective'].values)
        n_bins = np.prod(self.obj_archive.dims)

        obj_qd_per_bin = round(qd_obj/n_bins, 1)
        acq_qd_per_bin = round(qd_acq/n_bins, 1)
        obj_qd_per_elite = round(qd_obj/self.obj_archive.stats.num_elites, 1)
        acq_qd_per_elite = round(qd_acq/self.acq_archive.stats.num_elites, 1)

        # Print & Store Anytime Metrics
        anytime_metrics(self, acq_flag=True, iteration=iteration, current_eval_budget=current_eval_budget, consumed_obj_evals=consumed_obj_evals, obj_t0=obj_t0, obj_t1=obj_t1, obj_qd_per_elite=obj_qd_per_elite, obj_qd_per_bin=obj_qd_per_bin, n_new_obj_bins=n_new_obj_bins, n_new_obj_improvements=n_new_obj_improvements, n_new_obj_elites=n_new_obj_elites, percentage_convergence_errors=percentage_convergence_errors, percentage_new_obj_bins=percentage_new_obj_bins, percentage_obj_improvements=percentage_obj_improvements, percentage_new_obj_elites=percentage_new_obj_elites, total_new_obj_bins=total_new_obj_bins, total_obj_improvements=total_obj_improvements, total_new_obj_elites=total_new_obj_elites, total_convergence_errors=total_convergence_errors, convergence_errors=convergence_errors, n_new_target_bins=1337, n_new_target_improvements=1337, n_new_target_elites=1337, percentage_new_target_bins=percentage_new_acq_bins, percentage_target_improvements=percentage_acq_improvements, percentage_new_target_elites=percentage_new_acq_elites, total_new_target_bins=total_new_acq_bins, total_target_improvements=total_acq_improvements, total_new_target_elites=total_new_acq_elites, target_t0=acq_t0, target_t1=acq_t1, target_qd_per_bin=acq_qd_per_bin, target_qd_per_elite=acq_qd_per_elite)

        iteration += 1

        if iteration % 20 == 0:
            gc.collect()

    return


def run_random_sail(self: SailRun):
    """
    Extra evaluations are given if eval_pred_flag is True (see sail.py)
        if not eval_pred_flag, extra_evals = 0
    """

    eval_budget = ACQ_N_OBJ_EVALS + PRED_N_EVALS
    while(eval_budget >= BATCH_SIZE):

        ranges = np.array(SOL_VALUE_RANGE)

        def uniform_sample():
            uniform_sample = np.random.uniform(ranges[:, 0], ranges[:, 1], SOL_DIMENSION)
            return uniform_sample

        random_samples = np.array([uniform_sample() for i in range(BATCH_SIZE)])

        generate_parsec_coordinates(random_samples)
        convergence_errors, success_indices, obj_batch = xfoil(iterations=BATCH_SIZE)

        converged_samples = random_samples[success_indices]
        converged_behavior = random_samples[success_indices, 1:3]

        self.update_archive(converged_samples, obj_batch, converged_behavior, obj_flag=True)
        # update gp
        # render archives

        sol_array = np.vstack((sol_array, converged_samples)) # dtype=float64
        obj_array = np.vstack((obj_array, obj_batch.reshape(-1,1))) # dtype=float64

        eval_budget -= BATCH_SIZE

        print("\n\nAirfoil Convergence Errors: " + str(convergence_errors))
        print("Remaining Random Search Obj Evals: " + str(eval_budget) + "\n\n")

    return
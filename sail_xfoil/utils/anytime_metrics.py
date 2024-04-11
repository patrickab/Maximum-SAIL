import pandas
import numpy as np
from sail_run import SailRun

from config.config import Config
config = Config('config/config.ini')
BATCH_SIZE = config.BATCH_SIZE
ACQ_N_OBJ_EVALS = config.ACQ_N_OBJ_EVALS
ACQ_N_MAP_EVALS = config.ACQ_N_MAP_EVALS
PRED_N_OBJ_EVALS = config.PRED_N_OBJ_EVALS
PRED_N_MAP_EVALS = config.PRED_N_MAP_EVALS
INIT_N_ACQ_EVALS = config.INIT_N_ACQ_EVALS
PREDICTION_VERIFICATIONS = config.PREDICTION_VERIFICATIONS

def initialize_anytime_metrics(self: SailRun, acq_flag=False, pred_flag=False):

    anytime_metric_kwargs = {
        'iteration': 0,
        'current_eval_budget': 0,
        'consumed_obj_evals': 0,
        'consumed_target_evals': 0,

        'coverage': 0,

        'obj_qd' : 0,
        'obj_qd_per_bin' : 0,
        'obj_mean' : 0,
        
        'n_new_obj_bins': 0,
        'n_new_obj_elites': 0,
        'n_new_obj_improvements': 0,
        'best_obj_elite' : 0,

        'total_new_obj_bins': 0,
        'total_new_obj_elites': 0,
        'total_obj_improvements': 0,

        'percentage_new_obj_bins': 0,
        'percentage_new_obj_elites': 0,
        'percentage_new_obj_improvements': 0,

        'convergence_errors': 0,
        'total_convergence_errors': 0,
        'percentage_convergence_errors': 0,

    }

    return anytime_metric_kwargs


def calculate_anytime_metrics(self: SailRun, obj_t0, obj_t1, n_new_obj_elites, anytime_metric_kwargs, acq_flag=False, pred_flag=False):

    if acq_flag:
        i_obj_evals = BATCH_SIZE
    if pred_flag:
        i_obj_evals = PRED_N_OBJ_EVALS//PREDICTION_VERIFICATIONS

    target_archive = self.acq_archive if acq_flag else self.pred_archive
    n_bins = np.prod(target_archive.dims)

    anytime_metric_kwargs['iteration'] += 1

    anytime_metric_kwargs['current_eval_budget'] -= i_obj_evals

    anytime_metric_kwargs['consumed_obj_evals'] = anytime_metric_kwargs['total_eval_budget'] - anytime_metric_kwargs['current_eval_budget']
    
    anytime_metric_kwargs['obj_qd'] = round(self.obj_archive.stats.qd_score, 2)
    anytime_metric_kwargs['obj_mean'] = round(self.obj_archive.stats.obj_mean, 2)

    anytime_metric_kwargs['coverage'] = round(self.obj_archive.stats.coverage, 2)

    anytime_metric_kwargs['obj_t0'] = obj_t0
    anytime_metric_kwargs['obj_t1'] = obj_t1

    anytime_metric_kwargs['n_new_obj_bins'] = obj_t1 - obj_t0
    anytime_metric_kwargs['n_new_obj_elites'] = n_new_obj_elites
    anytime_metric_kwargs['n_new_obj_improvements'] = n_new_obj_elites - anytime_metric_kwargs['n_new_obj_bins']
    anytime_metric_kwargs['best_obj_elite'] = round(self.obj_archive.best_elite.objective, 2)

    anytime_metric_kwargs['total_new_obj_bins'] += anytime_metric_kwargs['n_new_obj_bins']
    anytime_metric_kwargs['total_new_obj_elites'] += n_new_obj_elites
    anytime_metric_kwargs['total_obj_improvements'] += anytime_metric_kwargs['n_new_obj_improvements']

    anytime_metric_kwargs['percentage_new_obj_bins'] = round((anytime_metric_kwargs['total_new_obj_bins']/anytime_metric_kwargs['consumed_obj_evals'])*100, 2)
    anytime_metric_kwargs['percentage_new_obj_elites'] = round((anytime_metric_kwargs['total_new_obj_elites']/anytime_metric_kwargs['consumed_obj_evals'])*100, 2)
    anytime_metric_kwargs['percentage_obj_improvements'] = round((anytime_metric_kwargs['total_obj_improvements']/anytime_metric_kwargs['consumed_obj_evals'])*100, 2)

    anytime_metric_kwargs['convergence_errors'] = self.convergence_errors
    anytime_metric_kwargs['total_convergence_errors'] += anytime_metric_kwargs['convergence_errors']
    anytime_metric_kwargs['percentage_convergence_errors'] = round((anytime_metric_kwargs['total_convergence_errors']/anytime_metric_kwargs['consumed_obj_evals'])*100, 2)

    return anytime_metric_kwargs


def store_anytime_metrics(self, anytime_metric_kwargs, acq_flag=False, pred_flag=False):

    if acq_flag:
        target = "Acq"
    if pred_flag:
        target = "Pred"

    anytime_dtypes = {
        'Remaining Obj Evals': int,
        'Consumed Obj Evals': int,
        'Coverage': float,
        'Obj Archive Size (before)': int,
        'Obj Archive Size (after)': int,
        'Obj QD (total)': float,
        'Obj QD (mean)': float,
        'New Convergence Errors': int,
        'New Obj Bins': int,
        'New Obj Improvements': int,
        'New Obj Elites': int,
        'Percentage Convergence Errors': float,
        'Percentage New Obj Bins': float,
        'Percentage Obj Improvements': float,
        'Percentage New Obj Elites': float,
        'Total Convergence Errors': int,
        'Total New Obj Bins': int,
        'Total Obj Improvements': int,
        'Total New Obj Elites': int,
        'Best Obj Elite Objective': float
    }

    anytime_metrics = pandas.DataFrame({c: pandas.Series(dtype=d) for c, d in anytime_dtypes.items()})
    pandas.options.display.float_format = '{:.2f}'.format

    iteration = anytime_metric_kwargs['iteration']

    current_eval_budget = anytime_metric_kwargs['current_eval_budget']
    consumed_obj_evals = anytime_metric_kwargs['consumed_obj_evals']
    coverage = anytime_metric_kwargs['coverage']
    obj_t0 = anytime_metric_kwargs['obj_t0']
    obj_t1 = anytime_metric_kwargs['obj_t1']
    obj_qd = anytime_metric_kwargs['obj_qd']
    obj_qd_mean = anytime_metric_kwargs['obj_mean']
    n_new_obj_bins = anytime_metric_kwargs['n_new_obj_bins']
    n_new_obj_improvements = anytime_metric_kwargs['n_new_obj_improvements']
    n_new_obj_elites = anytime_metric_kwargs['n_new_obj_elites']
    percentage_convergence_errors = anytime_metric_kwargs['percentage_convergence_errors']
    percentage_new_obj_bins = anytime_metric_kwargs['percentage_new_obj_bins']
    percentage_obj_improvements = anytime_metric_kwargs['percentage_obj_improvements']
    percentage_new_obj_elites = anytime_metric_kwargs['percentage_new_obj_elites']
    total_new_obj_bins = anytime_metric_kwargs['total_new_obj_bins']
    total_obj_improvements = anytime_metric_kwargs['total_obj_improvements']
    total_new_obj_elites = anytime_metric_kwargs['total_new_obj_elites']
    total_convergence_errors = anytime_metric_kwargs['total_convergence_errors']
    convergence_errors = anytime_metric_kwargs['convergence_errors']
    best_obj_elite = anytime_metric_kwargs['best_obj_elite']

    print(f"\n\nRemaining Obj Evals           : {current_eval_budget}")
    print(f"Consumed Obj Evals            : {consumed_obj_evals}\n")
    print(f"Obj Archive Size (before)     : {obj_t0}")
    print(f"Obj Archive Size  (after)     : {obj_t1}")
    print(f"Obj QD (per elite)            : {obj_qd_mean}")

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

    print(f'Best Obj Elite Objective     : {best_obj_elite}')

    anytime_data = [current_eval_budget, consumed_obj_evals, coverage, obj_t0, obj_t1, obj_qd, obj_qd_mean, convergence_errors, n_new_obj_bins, n_new_obj_improvements, n_new_obj_elites, percentage_convergence_errors, percentage_new_obj_bins, percentage_obj_improvements, percentage_new_obj_elites, total_convergence_errors, total_new_obj_bins, total_obj_improvements, total_new_obj_elites, best_obj_elite]
    anytime_metrics.loc[iteration] = anytime_data

    try:
        anytime_metrics.to_csv(f'{self.initial_seed}_{self.domain}_{target}_loop_anytime_metrics.csv', mode='a', header=False, index=True)
    except:
        anytime_metrics.to_csv(f'{self.initial_seed}_{self.domain}_{target}_loop_anytime_metrics.csv', header=True, index=True)
    anytime_metrics = pandas.DataFrame({c: pandas.Series(dtype=d) for c, d in anytime_dtypes.items()})

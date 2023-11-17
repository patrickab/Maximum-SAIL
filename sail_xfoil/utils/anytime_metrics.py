
### Foreign Scripts ###
import os
import pandas
import numpy as np

### Custom Scripts ###
from sail_runner import SailRun

### Configurable Variables ###
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

    if acq_flag:
        total_eval_budget = ACQ_N_OBJ_EVALS if self.pred_verific_flag else ACQ_N_OBJ_EVALS + PRED_N_OBJ_EVALS
        total_eval_budget += INIT_N_ACQ_EVALS if (self.random_init_flag or not self.custom_flag) else 0
        total_target_eval_budget = ACQ_N_MAP_EVALS * (total_eval_budget//BATCH_SIZE)
    if pred_flag:
        total_eval_budget = PRED_N_OBJ_EVALS
        total_target_eval_budget = PRED_N_MAP_EVALS

    anytime_metric_kwargs = {
        'iteration': 0,
        'consumed_obj_evals': 0,
        'consumed_target_evals': 0,

        'obj_qd' : 0,
        'obj_qd_per_bin' : 0,
        'obj_qd_per_elite' : 0,

        'target_qd' : 0,
        'target_qd_per_bin' : 0,
        'target_qd_per_elite' : 0,
        
        'n_new_obj_bins': 0,
        'n_new_obj_elites': 0,
        'n_new_obj_improvements': 0,
        'best_obj_elite' : 0,

        'n_new_target_bins': 0,
        'n_new_target_elites': 0,
        'n_new_target_improvements': 0,
        'best_target_elite' : 0,

        'total_new_obj_bins': 0,
        'total_new_obj_elites': 0,
        'total_obj_improvements': 0,

        'total_new_target_bins': 0,
        'total_new_target_elites': 0,
        'total_target_improvements': 0,

        'percentage_new_obj_bins': 0,
        'percentage_new_obj_elites': 0,
        'percentage_new_obj_improvements': 0,

        'percentage_new_target_bins': 0,
        'percentage_new_target_elites': 0,
        'percentage_new_target_improvements': 0,

        'convergence_errors': 0,
        'total_convergence_errors': 0,
        'percentage_convergence_errors': 0,

        'current_eval_budget': total_eval_budget,
        'current_target_eval_budget': total_target_eval_budget,

        'total_eval_budget': total_eval_budget,
        'total_target_eval_budget': total_target_eval_budget
    }

    return anytime_metric_kwargs


def calculate_anytime_metrics(self: SailRun, obj_t0, obj_t1, target_t0, target_t1, n_new_obj_elites, new_target_bin_elites, improved_target_elites, anytime_metric_kwargs, acq_flag=False, pred_flag=False):

    if acq_flag:
        i_obj_evals = BATCH_SIZE
        i_target_map_evals = ACQ_N_MAP_EVALS if self.acq_ucb_flag else ACQ_N_MAP_EVALS // 1.5
    if pred_flag:
        i_obj_evals = PRED_N_OBJ_EVALS//PREDICTION_VERIFICATIONS
        i_target_map_evals = PRED_N_MAP_EVALS//(PREDICTION_VERIFICATIONS+1)

    target_archive = self.acq_archive if acq_flag else self.pred_archive
    n_bins = np.prod(target_archive.dims)

    target_qd = sum(target_archive.as_pandas(include_solutions=True)['objective'].values)
    obj_qd = sum(self.obj_archive.as_pandas(include_solutions=True)['objective'].values)

    anytime_metric_kwargs['iteration'] += 1

    anytime_metric_kwargs['current_eval_budget'] -= i_obj_evals
    anytime_metric_kwargs['current_target_eval_budget'] -= i_target_map_evals

    anytime_metric_kwargs['consumed_obj_evals'] = anytime_metric_kwargs['total_eval_budget'] - anytime_metric_kwargs['current_eval_budget']
    anytime_metric_kwargs['consumed_target_evals'] = anytime_metric_kwargs['total_target_eval_budget'] - anytime_metric_kwargs['current_target_eval_budget']
    
    anytime_metric_kwargs['obj_qd'] = obj_qd
    anytime_metric_kwargs['obj_qd_per_bin'] = round(obj_qd/n_bins, 1)
    anytime_metric_kwargs['obj_qd_per_elite'] = round(obj_qd/self.obj_archive.stats.num_elites, 1)

    anytime_metric_kwargs['target_qd'] = target_qd
    anytime_metric_kwargs['target_qd_per_bin'] = round(anytime_metric_kwargs['target_qd']/n_bins, 1)
    anytime_metric_kwargs['target_qd_per_elite'] = round(anytime_metric_kwargs['target_qd']/target_archive.stats.num_elites, 1) if target_archive.stats.num_elites != 0 else 0

    anytime_metric_kwargs['obj_t0'] = obj_t0
    anytime_metric_kwargs['obj_t1'] = obj_t1
    anytime_metric_kwargs['n_new_obj_bins'] = obj_t1 - obj_t0
    anytime_metric_kwargs['n_new_obj_elites'] = n_new_obj_elites
    anytime_metric_kwargs['n_new_obj_improvements'] = n_new_obj_elites - anytime_metric_kwargs['n_new_obj_bins']
    anytime_metric_kwargs['best_obj_elite'] = self.obj_archive.best_elite.objective

    anytime_metric_kwargs['target_t0'] = target_t0
    anytime_metric_kwargs['target_t1'] = target_t1
    anytime_metric_kwargs['n_new_target_bins'] = target_t1 - target_t0
    anytime_metric_kwargs['n_new_target_improvements'] = -123 # ToDo (new_bin_elites & improved_elites refer to objective elites & cannot be used for this metric)
    anytime_metric_kwargs['n_new_target_elites'] = -123 # ToDo (new_bin_elites & improved_elites refer to objective elites & cannot be used for this metric)
    anytime_metric_kwargs['best_target_elite'] = target_archive.best_elite.objective

    anytime_metric_kwargs['total_new_obj_bins'] += anytime_metric_kwargs['n_new_obj_bins']
    anytime_metric_kwargs['total_new_obj_elites'] += n_new_obj_elites
    anytime_metric_kwargs['total_obj_improvements'] += anytime_metric_kwargs['n_new_obj_improvements']

    anytime_metric_kwargs['total_new_target_bins'] += anytime_metric_kwargs['n_new_target_bins']
    anytime_metric_kwargs['total_new_target_elites'] = -123 # ToDo (new_bin_elites & improved_elites refer to objective elites & cannot be used for this metric)
    anytime_metric_kwargs['total_target_improvements'] = -123 # ToDo (new_bin_elites & improved_elites refer to objective elites & cannot be used for this metric)

    anytime_metric_kwargs['percentage_new_obj_bins'] = (anytime_metric_kwargs['total_new_obj_bins']/anytime_metric_kwargs['consumed_obj_evals'])*100
    anytime_metric_kwargs['percentage_new_obj_elites'] = (anytime_metric_kwargs['total_new_obj_elites']/anytime_metric_kwargs['consumed_obj_evals'])*100
    anytime_metric_kwargs['percentage_obj_improvements'] = (anytime_metric_kwargs['total_obj_improvements']/anytime_metric_kwargs['consumed_obj_evals'])*100

    anytime_metric_kwargs['percentage_new_target_bins'] = (anytime_metric_kwargs['total_new_target_bins']/anytime_metric_kwargs['consumed_target_evals'])*100
    anytime_metric_kwargs['percentage_new_target_elites'] = -123 # ToDo (new_bin_elites & improved_elites refer to objective elites & cannot be used for this metric)
    anytime_metric_kwargs['percentage_target_improvements'] = -123 # ToDo (new_bin_elites & improved_elites refer to objective elites & cannot be used for this metric)

    anytime_metric_kwargs['convergence_errors'] = self.convergence_errors
    anytime_metric_kwargs['total_convergence_errors'] += anytime_metric_kwargs['convergence_errors']
    anytime_metric_kwargs['percentage_convergence_errors'] = (anytime_metric_kwargs['total_convergence_errors']/anytime_metric_kwargs['consumed_obj_evals'])*100

    return anytime_metric_kwargs


def store_anytime_metrics(self, anytime_metric_kwargs, acq_flag=False, pred_flag=False):

    if acq_flag:
        target = "Acq"
    if pred_flag:
        target = "Pred"

    anytime_dtypes = {'Remaining Obj Evals': int, 'Consumed Obj Evals': int, 'Obj Archive Size (before)': int, 'Obj Archive Size (after)': int, 'Obj QD (per bin)': float, 'Obj QD (per elite)': float, 'New Convergence Errors': int, 'New Obj Bins': int, 'New Obj Improvements': int, 'New Obj Elites': int, 'Percentage Convergence Errors': float, 'Percentage New Obj Bins': float, 'Percentage Obj Improvements': float, 'Percentage New Obj Elites': float, 'Total Convergence Errors': int, 'Total New Obj Bins': int, 'Total Obj Improvements': int, 'Total New Obj Elites': int, f'{target} Archive Size (before)': int, f'{target} Archive Size (after)': int, f'{target} QD (per bin)': float, f'{target} QD (per elite)': float, f'New {target} Bins': int, f'New {target} Improvements': int, f'New {target} Elites': int, f'Percentage New {target} Bins': float, f'Percentage {target} Improvements': float, f'Percentage New {target} Elites': float, f'Total New {target} Bins': int, f'Total {target} Improvements': int, f'Total New {target} Elites': int, f'Best {target} Elite Objective': float, f'Best Obj Elite Objective': float}
    anytime_metrics = pandas.DataFrame({c: pandas.Series(dtype=d) for c, d in anytime_dtypes.items()})
    pandas.options.display.float_format = '{:.2f}'.format

    iteration = anytime_metric_kwargs['iteration']

    current_eval_budget = anytime_metric_kwargs['current_eval_budget']
    consumed_obj_evals = anytime_metric_kwargs['consumed_obj_evals']
    obj_t0 = anytime_metric_kwargs['obj_t0']
    obj_t1 = anytime_metric_kwargs['obj_t1']
    obj_qd_per_elite = anytime_metric_kwargs['obj_qd_per_elite']
    obj_qd_per_bin = anytime_metric_kwargs['obj_qd_per_bin']
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
    n_new_target_bins = anytime_metric_kwargs['n_new_target_bins']
    n_new_target_improvements = anytime_metric_kwargs['n_new_target_improvements']
    n_new_target_elites = anytime_metric_kwargs['n_new_target_elites']
    percentage_new_target_bins = anytime_metric_kwargs['percentage_new_target_bins']
    percentage_target_improvements = anytime_metric_kwargs['percentage_target_improvements']
    percentage_new_target_elites = anytime_metric_kwargs['percentage_new_target_elites']
    total_new_target_bins = anytime_metric_kwargs['total_new_target_bins']
    total_target_improvements = anytime_metric_kwargs['total_target_improvements']
    total_new_target_elites = anytime_metric_kwargs['total_new_target_elites']
    target_t0 = anytime_metric_kwargs['target_t0']
    target_t1 = anytime_metric_kwargs['target_t1']
    n_new_target_elites = anytime_metric_kwargs['n_new_target_elites']
    target_qd_per_bin = anytime_metric_kwargs['target_qd_per_bin']
    n_new_target_bins = anytime_metric_kwargs['n_new_target_bins']
    target_qd_per_elite = anytime_metric_kwargs['target_qd_per_elite']
    best_obj_elite = anytime_metric_kwargs['best_obj_elite']
    best_target_elite = anytime_metric_kwargs['best_target_elite']

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
    print(f"Iteration Improvements       : {n_new_target_improvements}")
    print(f"Iteration New {target} Elites     : {n_new_target_elites}")
    print(f"Percentage New {target} Bins      : {percentage_new_target_bins:.1f}%")
    print(f"Percentage {target} Improvements  : {percentage_target_improvements:.1f}%")
    print(f"Percentage New {target} Elites    : {percentage_new_target_elites:.1f}%")
    print(f"Total New {target} Bins           : {total_new_target_bins}")
    print(f"Total Improvements           : {total_target_improvements}")
    print(f"Total New {target} Elites         : {total_new_target_elites}\n")

    print(f"{target} Archive Size (before)   : {target_t0}")
    print(f"{target} Archive Size  (after)   : {target_t1}")
    print(f"New {target} Elites              : {n_new_target_elites}\n")
    print(f"{target} QD (per bin)            : {target_qd_per_bin}")
    print(f"New {target} Bins                : {n_new_target_bins}")
    print(f"Mean {target} QD                 : {target_qd_per_elite}\n")

    print(f'Best Obj Elite Objective     : {best_obj_elite}')
    print(f"Best {target} Elite Objective     : {best_target_elite}\n")

    anytime_data = [current_eval_budget, consumed_obj_evals, obj_t0, obj_t1, obj_qd_per_bin, obj_qd_per_elite, convergence_errors, n_new_obj_bins, n_new_obj_improvements, n_new_obj_elites, percentage_convergence_errors, percentage_new_obj_bins, percentage_obj_improvements, percentage_new_obj_elites, total_convergence_errors, total_new_obj_bins, total_obj_improvements, total_new_obj_elites, target_t0, target_t1, target_qd_per_bin, target_qd_per_elite, n_new_target_bins, n_new_target_improvements, n_new_target_elites, percentage_new_target_bins, percentage_target_improvements, percentage_new_target_elites, total_new_target_bins, total_target_improvements, total_new_target_elites, best_target_elite, best_obj_elite]
    anytime_metrics.loc[iteration] = anytime_data

    if iteration % 2 == 0:
        # Save Anytime Metrics to CSV. If no csv is saved create new csv, else append to existing csv
        target = "acq" if acq_flag else "pred"
        try:
            anytime_metrics.to_csv(f'{self.initial_seed}_{self.domain}_{target}_loop_anytime_metrics.csv', mode='a', header=False, index=True)
        except:
            anytime_metrics.to_csv(f'{self.initial_seed}_{self.domain}_{target}_loop_anytime_metrics.csv', header=True, index=True)
        anytime_metrics = pandas.DataFrame({c: pandas.Series(dtype=d) for c, d in anytime_dtypes.items()})
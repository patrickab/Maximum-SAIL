###### Import Foreign Scripts ######
from chaospy import create_sobol_samples
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from gp.predict_objective import predict_objective
import gc
import numpy as np
import pandas
import logging
import os

### Configurable Variables ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), 'config', 'config.ini'))
PREDICTION_VERIFICATIONS = config.PREDICTION_VERIFICATIONS
MAX_PRED_VERIFICATION = config.MAX_PRED_VERIFICATION
ACQ_N_OBJ_EVALS = config.ACQ_N_OBJ_EVALS
ACQ_N_MAP_EVALS = config.ACQ_N_MAP_EVALS
INIT_N_EVALS = config.INIT_N_EVALS
PRED_N_EVALS = config.PRED_N_EVALS
BATCH_SIZE = config.BATCH_SIZE

n_obj_evals = INIT_N_EVALS + ACQ_N_OBJ_EVALS + MAX_PRED_VERIFICATION
CSV_BUFFERSIZE = (n_obj_evals/BATCH_SIZE) / 8

###### Import Custom Scripts ######

from utils.utils import eval_xfoil_loop, scale_samples
from utils.anytime_archive_visualizer import anytime_archive_visualizer
from utils.pprint_nd import pprint

from xfoil.generate_airfoils import generate_parsec_coordinates
from xfoil.simulate_airfoils import xfoil

from acq_functions.acq_ucb import acq_ucb
from gp.fit_gp_model import fit_gp_model
from map_elites import map_elites

from sail_runner import SailRun

def run_custom_sail(self: SailRun):
    """
    Extra evaluations are given if eval_pred_flag is True (see sail.py)
        if not eval_pred_flag, extra_evals = 0
    """

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

    anytime_dtypes = {'Remaining Obj Evals': int, 'Consumed Obj Evals': int, 'Obj Archive Size (before)': int, 'Obj Archive Size (after)': int, 'Obj QD (per bin)': float, 'Obj QD (per elite)': float, 'New Convergence Errors': int, 'New Obj Bins': int, 'New Obj Improvements': int, 'New Obj Elites': int, 'Percentage Convergence Errors': float, 'Percentage New Obj Bins': float, 'Percentage Obj Improvements': float, 'Percentage New Obj Elites': float, 'Total Convergence Errors': int, 'Total New Obj Bins': int, 'Total Obj Improvements': int, 'Total New Obj Elites': int, 'Acq Archive Size (before)': int, 'Acq Archive Size (after)': int, 'Acq QD (per bin)': float, 'Acq QD (per elite)': float, 'New Acq Bins': int, 'New Acq Improvements': int, 'New Acq Elites': int, 'Percentage New Acq Bins': float, 'Percentage Acq Improvements': float, 'Percentage New Acq Elites': float, 'Total New Acq Bins': int, 'Total Acq Improvements': int, 'Total New Acq Elites': int}
    anytime_metrics = pandas.DataFrame({c: pandas.Series(dtype=d) for c, d in anytime_dtypes.items()})
    pandas.options.display.float_format = '{:.2f}'.format

    total_eval_budget = ACQ_N_OBJ_EVALS if self.pred_verific_flag else ACQ_N_OBJ_EVALS + MAX_PRED_VERIFICATION # if no budget for prediction verification is given, add MAX_PRED_VERIFICATION to ACQ_N_MAP_EVALS to ensure equal number of evaluations
    total_acq_eval_budget = ACQ_N_MAP_EVALS * (ACQ_N_OBJ_EVALS//BATCH_SIZE)
    current_eval_budget = total_eval_budget
    current_acq_eval_budget = total_acq_eval_budget

    while(current_eval_budget >= BATCH_SIZE):

        old_elites = np.array([(  elite.solution,     elite.index,     elite.objective,       elite.measures) for elite in self.obj_archive],
                        dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])

        # calculate & evaluate elites, that maximize acquisition values
        # update obj_archive and gp_model inside eval_max_obj_improvement()
        acq_archive, new_acq_elite_archive, acq_t0, acq_t1 = map_elites(self, target_function=acq_ucb, acq_flag=True)
        improved_elites, new_bin_elites = maximize_improvement(new_elite_archive=new_acq_elite_archive, old_elites=old_elites) 
        evaluate_max_improvement(self, improved_elites=improved_elites, new_bin_elites=new_bin_elites, old_elites=old_elites, target_function=acq_ucb, acq_flag=True)

        current_eval_budget -= BATCH_SIZE
        current_acq_eval_budget -= ACQ_N_MAP_EVALS
        consumed_obj_evals = total_eval_budget - current_eval_budget
        consumed_acq_evals = total_acq_eval_budget - current_acq_eval_budget
        
        # Count newly discovered elites
        obj_t0 = self.obj_t0
        obj_t1 = self.obj_t1
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

        # Print Anytime Metrics
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

        print(f"Iteration New Acq Bins        : {n_new_acq_bins}")
        print(f"Iteration Improvements        : {n_new_acq_improvements}")
        print(f"Iteration New Acq Elites      : {n_new_acq_elites}")
        print(f"Percentage New Acq Bins       : {percentage_new_acq_bins:.1f}%")
        print(f"Percentage Acq Improvements   : {percentage_acq_improvements:.1f}%")
        print(f"Percentage New Acq Elites     : {percentage_new_acq_elites:.1f}%")
        print(f"Total New Acq Bins            : {total_new_acq_bins}")
        print(f"Total Improvements            : {total_acq_improvements}")
        print(f"Total New Acq Elites          : {total_new_acq_elites}\n")

        print(f"Acq Archive Size (before)    : {acq_t0}")
        print(f"Acq Archive Size  (after)    : {acq_t1}")
        print(f"New Acq Elites               : {n_new_acq_elites}\n")

        print(f"Acq QD (per bin)             : {acq_qd_per_bin}")
        print(f"New Acq Bins                 : {n_new_acq_bins}")
        print(f"Mean Acq QD                  : {acq_qd_per_elite}\n")


        # ToDo: mean improvements (new_converged_elites - old_corresponding_elites) / n_converged_elites 
        # ToDo: print new_converged_elites next to old_corresponding_elites
        # old_corresponding_elites -> right join by index
        # print("Mean Obj Improvement: {:.1f}".format(mean_obj_improvement))
        old_elite_obj = old_elites['objective']

        # Print Anytime Metrics
        anytime_data = [current_eval_budget, consumed_obj_evals, obj_t0, obj_t1, obj_qd_per_bin, obj_qd_per_elite, convergence_errors, n_new_obj_bins, n_new_obj_improvements, n_new_obj_elites, percentage_convergence_errors, percentage_new_obj_bins, percentage_obj_improvements, percentage_new_obj_elites, total_convergence_errors, total_new_obj_bins, total_obj_improvements, total_new_obj_elites, acq_t0, acq_t1, acq_qd_per_bin, acq_qd_per_elite, n_new_acq_bins, n_new_acq_improvements, n_new_acq_elites, percentage_new_acq_bins, percentage_acq_improvements, percentage_new_acq_elites, total_new_acq_bins, total_acq_improvements, total_new_acq_elites]
        anytime_metrics.loc[iteration] = anytime_data

        if iteration % 2 == 0:
            # Save Anytime Metrics to CSV. If no csv is saved create new csv, else append to existing csv
            try:
                anytime_metrics.to_csv(f'{self.initial_seed}_{self.domain}_acq_loop_anytime_metrics.csv', mode='a', header=False, index=True)
            except:
                anytime_metrics.to_csv(f'{self.initial_seed}_{self.domain}_acq_loop_anytime_metrics.csv', header=True, index=True)
            anytime_metrics = pandas.DataFrame({c: pandas.Series(dtype=d) for c, d in anytime_dtypes.items()})
            
        iteration += 1

    if iteration % 20 == 0:
        gc.collect()

    return


def maximize_improvement(new_elite_archive: GridArchive, old_elites: np.ndarray):
    """
    - extracts all elites from new_elite_archive
    - orders them by objective improvement

    Input: 
        (Grid_Archive): new_elite_archive
        (np_ndarray): old elites         
            -> old_elites = np.array([(elite.solution, elite.index, elite.objective, elite.measures) for elite in obj_archive], dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)]))
    """
    # ToDo: Verify

    elites = np.array([(elite.solution, elite.index, elite.objective, elite.measures) for elite in new_elite_archive], dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])
    elites = elites[np.argsort(elites['index'])]
    
    # Seperate improved elites (niche compete) from new elites (new niches)
    is_improved_new_elite = np.isin(elites['index'], old_elites['index'])

    improved_elites = elites[is_improved_new_elite]
    new_bin_elites   = elites[~is_improved_new_elite]

    # Sort by index
    improved_elites = improved_elites[np.argsort(improved_elites['index'])]
    new_bin_elites   = new_bin_elites[np.argsort(new_bin_elites['index'])]

    # Select old elites that have been improved
    is_improved_old_elite = np.isin(old_elites['index'], improved_elites['index'])
    old_elites_improved   = old_elites[is_improved_old_elite]
    old_elites_improved   = old_elites_improved[np.argsort(old_elites_improved['index'])]

    objective_improvement = improved_elites['objective'] - old_elites_improved['objective']

    # Pack into one data structure
    improved_elites = np.array(list(zip(
        improved_elites['solution'], improved_elites['objective'],          objective_improvement , improved_elites['behavior'])), 
        dtype=[        ('solution', object),        ('objective', float), ('objective_improvement', float),        ('behavior', object)])
    # Sort & flip to ensure descending order
    improved_elites = improved_elites[np.argsort(improved_elites['objective_improvement'])]
    improved_elites = np.flip(improved_elites)
    
    new_bin_elites = np.array(list(zip(
        new_bin_elites['solution'], new_bin_elites['objective'],        new_bin_elites['objective'], new_bin_elites['behavior'])), 
           dtype=[('solution', object),   ('objective', float),('objective_improvement', float),   ('behavior', object)])
    
    new_bin_elites      = new_bin_elites[np.argsort(new_bin_elites['objective_improvement'])]
    new_bin_elites      = np.flip(new_bin_elites)

    return improved_elites, new_bin_elites


def evaluate_max_improvement(self: SailRun, improved_elites, new_bin_elites, old_elites, target_function, acq_flag=False, pred_flag=False):

    """
    Evaluates elites, that present the maximum improvement regarding their respective objective
                                                             (acquisition or prediction values)

    IMPORTANT: improved_elites, new elites are to be exected in a specific np.ndarray datastructure (see)

    Input:
        "improved_elites": Elites sorted in descending order (niche competition winners)
        "old_elites"   :   Elites sorted in descending order (niche competition losers)
        "new_bin_elites"   :   Elites sorted in descending order (new bin discoveries)

        "target_function"     :   Obj function for map_elites

        "new_elite_archive": (((check if necessary)))
    """

    if not acq_flag and not pred_flag:
        raise ValueError("Evaluate Max Improvement: Either acq_flag or pred_flag has to be True")

    def ensure_n_samples(improved_elites, new_bin_elites, acq_flag, pred_flag):

        """
        Ensures that the appropiate number of samples is evaluated, by reevaluating Map Elites (if necessary)
        After 4 extra evaluations, the function returns the best elites found so far to avoid infinite loops
        """

        target = "Acq" if acq_flag else "Pred"

        if acq_flag:
            n_samples = BATCH_SIZE
        if pred_flag and self.pred_verific_flag:
            n_samples = MAX_PRED_VERIFICATION//PREDICTION_VERIFICATIONS
        if pred_flag and not self.pred_verific_flag:
            raise ValueError("Maximize Improvement: Prediction Flag is True, but Prediction Verification Flag is False")

        # If enough elites have been sampled, return
        print("\nn_samples: " + str(n_samples))
        print("n improvements: " + str(improved_elites.shape[0]+new_bin_elites.shape[0]))
        if n_samples < improved_elites.shape[0] + new_bin_elites.shape[0]:
            print(f'Enough {target} Improvements: Returning')
            return improved_elites, new_bin_elites, n_samples

        # Sample more elites & add improved elites + new bin elites to target_archive
        i_target_archive, i_new_elite_archive, _, _ = map_elites(self, target_function=target_function, acq_flag=acq_flag, pred_flag=pred_flag)
        i_improved_elites, i_new_bin_elites = maximize_improvement(i_new_elite_archive, old_elites)
        n_improvements = i_new_elite_archive.stats.num_elites
        iteration = 0

        # Re-enter MAP-Elites (acq/obj) up to 2 times if necessary 
        while n_improvements < n_samples and iteration <= 5:

            iteration += 1

            print(f'\n\n### Not enough {target} Improvements: Re-entering {target} Loop###\n\n')
            i_target_archive, i_new_elite_archive, _, _ = map_elites(self, target_function=target_function, new_elite_archive=i_new_elite_archive, acq_flag=acq_flag, pred_flag=pred_flag)
            n_improvements = i_new_elite_archive.stats.num_elites
            print("n_improvements: " + str(n_improvements))

        # Enough samples have been found, or Loop has been re-entered twice
        # Proceed to sample selection
        i_improved_elites, i_new_bin_elites = maximize_improvement(i_new_elite_archive, old_elites)
        return i_improved_elites, i_new_bin_elites, n_samples


    
    def select_samples(improved_elites, new_bin_elites, n_samples):
        """Selects samples based on exploration behavior defined in the class constructor"""

        if self.explore_flag:
            # Evaluate new_elites first
            candidate_elites = np.concatenate((new_bin_elites, improved_elites), axis=0)
            n_candidate_elites = candidate_elites[:n_samples]
        if self.greedy_flag:
            # Evaluate only maximum improvement, regardeless of new/old bin
            candidate_elites = np.concatenate((improved_elites, new_bin_elites), axis=0)
            # sort by objective improvement in reversed order
            candidate_elites = candidate_elites[np.argsort(candidate_elites['objective_improvement'])][::-1]
            n_candidate_elites = candidate_elites[:n_samples]
        if self.hybrid_flag:
            # Evenly balance sampling of best new_bin_elites & best improved_elites
            n_new_bin_elites = new_bin_elites.shape[0]
            n_improved_elites = improved_elites.shape[0]
            if n_new_bin_elites >= n_samples//2 and n_improved_elites >= n_samples//2:
                n_candidate_elites = np.concatenate((improved_elites[:n_samples//2], new_bin_elites[:n_samples//2]), axis=0)
            else:
                if n_new_bin_elites < n_samples//2:
                    candidate_elites = np.concatenate((new_bin_elites, improved_elites), axis=0)
                    n_candidate_elites = candidate_elites[:n_samples]
                else:
                    candidate_elites = np.concatenate((improved_elites, new_bin_elites), axis=0)
                    n_candidate_elites = candidate_elites[:n_samples]

        print("\n\nSolutions to be evaluated next: ")
        target_objective = n_candidate_elites['objective']
        target_objective_improvement = n_candidate_elites['objective_improvement']
        pprint(target_objective, target_objective_improvement)

        return n_candidate_elites

    i_improved_elites, i_new_bin_elites, n_samples = ensure_n_samples(improved_elites=improved_elites, new_bin_elites=new_bin_elites, acq_flag=acq_flag, pred_flag=pred_flag)
    candidate_elites = select_samples(improved_elites=i_improved_elites, new_bin_elites=i_new_bin_elites, n_samples=n_samples)

    # fill obj archive inside eval_xfoil_loop() & update acq_archive/pred_archive with new obj solutions
    eval_xfoil_loop(self, candidate_sol=candidate_elites['solution'], candidate_acq_or_pred=candidate_elites['objective'])


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

    anytime_columns = ['Iteration', 'Obj QD (per elite)', 'Obj QD (per bin)', 
                                    'Acq QD (per elite)', 'Acq QD (per bin)',
                                    'Percentage New Obj Elites', 'Total New Obj Elites', 'Iteration New Obj Elites',
                                    'Percentage New Acq Elites', 'Total New Acq Elites', 'Iteration New Acq Elites',
                                    'Percentage Obj Improvements', 'Total Obj Improvements', 'Iteration Obj Improvements',
                                    'Percentage Acq Improvements', 'Total Acq Improvements', 'Iteration Acq Improvements',
                                    'Percentage New Obj Bins', 'Total New Obj Bins', 'Iteration New Obj Bins',
                                    'Percentage New Acq Bins', 'Total New Acq Bins', 'Iteration New Acq Bins',
                                    'Percentage Convergence Errors', 'Total Convergence Errors', 'Iteration Convergence Errors']

    anytime_metrics = pandas.DataFrame(columns=anytime_columns)
    total_eval_budget = MAX_PRED_VERIFICATION
    total_pred_eval_budget = PRED_N_EVALS
    current_eval_budget = total_eval_budget
    current_pred_eval_budget = total_pred_eval_budget
    iter_evals = MAX_PRED_VERIFICATION//(PREDICTION_VERIFICATIONS)

    while(current_eval_budget >= iter_evals):

        old_elites = np.array([(  elite.solution,     elite.index,     elite.objective,       elite.measures) for elite in self.obj_archive],
                        dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])

        pred_archive, new_pred_elite_archive, pred_t0, pred_t1 = map_elites(self, target_function=predict_objective, pred_flag=True)
        improved_elites, new_bin_elites = maximize_improvement(new_elite_archive=new_pred_elite_archive, old_elites=old_elites) 
        evaluate_max_improvement(self, improved_elites=improved_elites, new_bin_elites=new_bin_elites, old_elites=old_elites, target_function=predict_objective, pred_flag=True)

        self.visualize_archive(archive=self.pred_archive, pred_flag=True)

        current_eval_budget -= iter_evals
        current_pred_eval_budget -= PRED_N_EVALS//(MAX_PRED_VERIFICATION+1) # +1 because after the last prediction verification we predict once more

        consumed_obj_evals = total_eval_budget - current_eval_budget
        consumed_pred_evals = total_pred_eval_budget - current_pred_eval_budget
        
        # Count newly discovered elites
        obj_t0 = self.obj_t0
        obj_t1 = self.obj_t1
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
        percentage_obj_improvements   = (total_obj_improvements/consumed_obj_evals)*100
        percentage_pred_improvements   = (total_pred_improvements/consumed_pred_evals)*100

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

        # ToDo: mean improvements (new_converged_elites - old_corresponding_elites) / n_converged_elites 
        # ToDo: print new_converged_elites next to old_corresponding_elites
        # old_corresponding_elites -> right join by index
        # print("Mean Obj Improvement: {:.1f}".format(mean_obj_improvement))
        old_elite_obj = old_elites['objective']

        # Print Anytime Metrics
        print(f"\n\nRemaining Obj Evals          : {current_eval_budget}")
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
        print(f"Total New Obj Elites          : {total_new_obj_elites}")
        print(f"Total Consumed Evals          : {consumed_obj_evals}\n")

        print(f"Total Convergence Errors      : {total_convergence_errors}")
        print(f"Iteration Convergence Errors  : {convergence_errors}\n")

        print(f"Iteration New Pred Bins        : {n_new_pred_bins}")
        print(f"Iteration Improvements        : {n_new_pred_improvements}")
        print(f"Iteration New Pred Elites      : {n_new_pred_elites}")
        print(f"Percentage New Pred Bins       : {percentage_new_pred_bins:.1f}%")
        print(f"Percentage Pred Improvements   : {percentage_pred_improvements:.1f}%")
        print(f"Percentage New Pred Elites     : {percentage_new_pred_elites:.1f}%")
        print(f"Total New Pred Bins            : {total_new_pred_bins}")
        print(f"Total Improvements            : {total_pred_improvements}")
        print(f"Total New Pred Elites          : {total_new_pred_elites}\n")

        print(f"Pred Archive Size (before)    : {pred_t0}")
        print(f"Pred Archive Size  (after)    : {pred_t1}")
        print(f"New Pred Elites               : {n_new_pred_elites}\n")

        print(f"Pred QD (per bin)             : {pred_qd_per_bin}")
        print(f"New Pred Bins                 : {n_new_pred_bins}")
        print(f"Mean Pred QD                  : {pred_qd_per_elite}\n")


        # Store Anytime Metrics in Pandas Dataframe
        anytime_data = [iteration, obj_qd_per_elite, obj_qd_per_bin,
                                   pred_qd_per_elite, pred_qd_per_bin,
                                   percentage_new_obj_elites, total_new_obj_elites, n_new_obj_elites,
                                   percentage_new_pred_elites, total_new_pred_elites, n_new_pred_elites,
                                   percentage_obj_improvements, total_obj_improvements, n_new_obj_elites,
                                   percentage_pred_improvements, total_pred_improvements, n_new_pred_elites,
                                   percentage_new_obj_bins, total_new_obj_bins, n_new_obj_bins,
                                   percentage_new_pred_bins, total_new_pred_bins, n_new_pred_bins,
                                   percentage_convergence_errors, total_convergence_errors, convergence_errors,]
        anytime_metrics.loc[len(anytime_metrics)] = anytime_data

        if iteration % 20 == 0:
            # Save Anytime Metrics to CSV. If no csv is saved create new csv, else append to existing csv
            try:
                anytime_metrics.to_csv(f'{self.initial_seed}_{self.domain}_pred_loop_anytime_metrics.csv', mode='a', header=False, index=False)
            except:
                anytime_metrics.to_csv(f'{self.initial_seed}_{self.domain}_pred_loop_anytime_metrics.csv', index=False)
            # Reset to empty dataframe
            anytime_metrics = pandas.DataFrame(columns= anytime_columns)
            
        iteration += 1



    if iteration % 20 == 0:
        gc.collect()

    pred_archive, new_pred_elite_archive, pred_t0, pred_t1 = map_elites(self, target_function=predict_objective, pred_flag=True)


    return pred_archive

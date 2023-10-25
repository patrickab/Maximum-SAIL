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
TEST_RUNS = config.TEST_RUNS
BATCH_SIZE = config.BATCH_SIZE
INIT_N_EVALS = config.INIT_N_EVALS
PRED_N_EVALS = config.PRED_N_EVALS
SIGMA_EMITTER = config.SIGMA_EMITTER
SOL_DIMENSION = config.SOL_DIMENSION
OBJ_DIMENSION = config.OBJ_DIMENSION
BHV_DIMENSION = config.BHV_DIMENSION
ACQ_N_OBJ_EVALS = config.ACQ_N_OBJ_EVALS
ACQ_N_MAP_EVALS = config.ACQ_N_MAP_EVALS
BHV_NUMBER_BINS = config.BHV_NUMBER_BINS
BHV_VALUE_RANGE = config.BHV_VALUE_RANGE
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SIGMA_PRED_EMITTER = config.SIGMA_PRED_EMITTER

MAX_PRED_VERIFICATION = config.MAX_PRED_VERIFICATION
PREDICTION_VERIFICATIONS = config.PREDICTION_VERIFICATIONS

total_obj_evals = INIT_N_EVALS + ACQ_N_OBJ_EVALS + MAX_PRED_VERIFICATION

###### Import Custom Scripts ######

from utils.utils import eval_xfoil_loop, scale_samples
from utils.anytime_archive_visualizer import anytime_archive_visualizer
from utils.pprint_nd import pprint

from xfoil.generate_airfoils import generate_parsec_coordinates
from xfoil.simulate_airfoils import xfoil

from acq_functions.acq_ucb import acq_ucb
from gp.fit_gp_model import fit_gp_model
from map_elites import map_elites

class SailRun:

    def __init__(self, initial_seed, sail_vanilla_flag=False, sail_custom_flag=False, sail_random_flag=False, pred_verific_flag=False, greedy_flag=False, explore_flag=False, hybrid_flag=False):

        """
        Initialize a SAIL Run.

        Args:
            initial_seed (int): The initial seed value.
            vanilla_flag (bool): Flag for vanilla SAIL.
            custom_flag (bool): Flag for custom SAIL.
            random_flag (bool): Flag for random SAIL.
            pred_verific_flag (bool): Flag for prediction verification.
            extra_evals (int): The number of extra evaluations.
        """

        # read into logger & use properly for DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialize SAIL Run")

        self.obj_current_iteration = 1
        self.new_current_iteration = 1
        self.acq_current_iteration = 1
        self.pred_current_iteration = 1

        if sail_custom_flag:
            self.domain = "custom"
        if sail_random_flag:
            self.domain = "random"
        if sail_vanilla_flag:
            self.domain = "vanilla"
        if greedy_flag:
            self.domain = self.domain + "_greedy"
        if explore_flag:
            self.domain = self.domain + "_explore"
        if hybrid_flag:
            self.domain = self.domain + "_hybrid"
        if pred_verific_flag:
            self.domain = self.domain + "_prediction_verification"
        if greedy_flag and explore_flag:
            raise ValueError("Greedy and Explore Flags cannot both be True")
        if pred_verific_flag and not (greedy_flag or explore_flag or hybrid_flag):
            raise ValueError("Prediction Verification Flag requires Greedy or Explore Flag to be True")

        # stores new solutions from reevaluate_archive()
        self.new_sol = np.empty((0, SOL_DIMENSION))
        self.new_obj = np.empty((0, OBJ_DIMENSION))
        self.new_bhv = np.empty((0, BHV_DIMENSION))

        self.initial_seed = initial_seed
        self.current_seed = initial_seed

        self.convergence_errors = 0

        self.custom_flag = sail_custom_flag
        self.vanilla_flag = sail_vanilla_flag
        self.random_flag = sail_random_flag

        self.greedy_flag = greedy_flag
        self.explore_flag = explore_flag
        self.hybrid_flag = hybrid_flag
        self.pred_verific_flag = pred_verific_flag

        self.sol_array = np.empty((0, SOL_DIMENSION))
        self.obj_array = np.empty((0, OBJ_DIMENSION))

        self.obj_archive, self.acq_archive, self.pred_archive, self.new_archive, self.evaluated_predictions_archive, self.prediction_error_archive = self.define_archives(initial_seed)

        print("\n\n\nInitialize SAIL Run")
        print(f"Domain: {self.domain}")
        print(f"Initial Seed: {self.initial_seed}")    
        print(f"Initialize Archive [...]")
        total_errors = 0

        samples = create_sobol_samples(order=INIT_N_EVALS, dim=len(SOL_VALUE_RANGE), seed=self.current_seed+5)
        samples = samples.T
        scale_samples(samples) # sobol samples are between [0;1]
        eval_xfoil_loop(self, candidate_sol=samples) # fill obj archive inside eval_xfoil_loop() & update+render acq_archive
        print("[...] Terminate init_archive()\n")

    
    def update_gp_data(self, new_solutions, new_objectives):

        print(f"Update GP Data [...]\n")
        n_new = new_solutions.shape[0]
        n_old = self.sol_array.shape[0]
        n_expected = n_old + n_new 
        # np.vstack x and y for bulletproof functionality
        new_solutions = np.vstack(new_solutions) if new_solutions.shape[0] != 0 else new_solutions
        new_objectives = np.vstack(new_objectives) if new_solutions.shape[0] != 0 else new_objectives
        self.sol_array = np.vstack((self.sol_array, new_solutions))
        self.obj_array = np.vstack((self.obj_array, new_objectives))
        n_resulted = self.sol_array.shape[0]

        if n_resulted != n_expected:
            raise ValueError("GP Data Update Error")

    def update_gp_model(self, new_solutions=None, new_objectives=None):

        if new_solutions is None and new_objectives is None:
            self.gp_model = fit_gp_model(self.sol_array, self.obj_array)
            return

        # np.vstack x and y for bulletproof functionality
        new_solutions = np.vstack(new_solutions) if new_solutions.shape[0] != 0 else new_solutions
        new_objectives = np.vstack(new_objectives) if new_solutions.shape[0] != 0 else new_objectives

        self.sol_array = np.vstack((self.sol_array, new_solutions))
        self.obj_array = np.vstack((self.obj_array, new_objectives))
        print("sol array shape" + str(self.sol_array.shape))
        print("obj array shape" + str(self.obj_array.shape))

        self.gp_model = fit_gp_model(self.sol_array, self.obj_array)

    def update_seed(self):
        self.current_seed += TEST_RUNS
        return self.current_seed

    def visualize_archive(self, archive, obj_flag=False, acq_flag=False, pred_flag=False, new_flag=False):
        anytime_archive_visualizer(self, archive=archive, obj_flag=obj_flag, acq_flag=acq_flag, pred_flag=pred_flag, new_flag=new_flag)
        if obj_flag:
            self.obj_current_iteration += 1
        if new_flag:
            self.new_current_iteration += 1
        if acq_flag:
            self.acq_current_iteration += 1
        if pred_flag:
            self.pred_current_iteration += 1

    def update_archive(self, candidate_sol=None, candidate_obj=None, candidate_bhv=None, obj_flag=False, acq_flag=False, pred_flag=False, evaluate_prediction_archive=False):
        """"
        Input:
            Option 1: Call with archive & archive flag
            Option 2: Call with candidate_sol, candidate_obj, candidate_bhv & archive flag
        """            

        if np.sum([obj_flag, acq_flag, pred_flag, evaluate_prediction_archive]) != 1:
            raise ValueError("Update Archive: Exactly one flag is supposed to be true - update archives seperately")
        
        if obj_flag:

            candidate_obj = candidate_obj.ravel()

            status_vector, _ = self.obj_archive.add(candidate_sol, candidate_obj, candidate_bhv)
            non_0_status_indices = np.where(status_vector != 0)[0]
            self.new_sol = candidate_sol[non_0_status_indices]
            self.new_obj = candidate_obj[non_0_status_indices]
            self.new_bhv = candidate_bhv[non_0_status_indices]

            self.new_archive.clear()
            self.new_archive.add(self.new_sol, self.new_obj, self.new_bhv)

            return
        
        if evaluate_prediction_archive:
            self.evaluated_predictions_archive.add(candidate_sol, candidate_obj, candidate_bhv)
            return
        
        if candidate_obj != None:
            raise ValueError("Update Archive: candidate_obj is supposed to be None when updating acq/pred archive")

        if acq_flag:
            candidate_acq = acq_ucb(genomes=candidate_sol, gp_model=self.gp_model) if candidate_sol.shape[0] != 0 else None
            self.acq_archive.add(candidate_sol, candidate_acq, candidate_bhv) if candidate_sol.shape[0] != 0 else None
        if pred_flag:
            candidate_pred = predict_objective(genomes=candidate_sol, gp_model=self.gp_model) if candidate_sol.shape[0] != 0 else None
            self.pred_archive.add(candidate_sol, candidate_pred, candidate_bhv) if candidate_sol.shape[0] != 0 else None
    

    def define_archives(self, seed):

        obj_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = -1
        )

        acq_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = -1
        )

        pred_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = -1
        )

        # Used for visualizing new elites (improved + new bin discoveries)
        new_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = -1
        )

        # Used for evaluating quality of results
        evaluated_predictions_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = -1
        )

        # Used for visualizing prediction errors
        prediction_error_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = -1
        )

        return obj_archive, acq_archive, pred_archive, new_archive, evaluated_predictions_archive, prediction_error_archive


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

    anytime_columns = ['Iteration', 'Obj QD (per elite)', 'Obj QD (per bin)', 
                                    'Acq QD (per elite)', 'Acq QD (per bin)',
                                    'Percentage New Obj Elites', 'Total New Obj Elites', 'Iteration New Obj Elites',
                                    'Percentage New Acq Elites', 'Total New Acq Elites', 'Iteration New Acq Elites',
                                    'Percentage Obj Improvements', 'Total Obj Improvements', 'Iteration Obj Improvements',
                                    'Percentage Acq Improvements', 'Total Acq Improvements', 'Iteration Acq Improvements',
                                    'Percentage New Obj Bins', 'Total New Obj Bins', 'Iteration New Obj Bins',
                                    'Percentage New Acq Bins', 'Total New Acq Bins', 'Iteration New Acq Bins',
                                    'Percentage Convergence Errors', 'Total Convergence Errors', 'Iteration Convergence Errors']

    anytime_metrics = pandas.DataFrame(columns= anytime_columns)

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

        # ToDo: mean improvements (new_converged_elites - old_corresponding_elites) / n_converged_elites 
        # ToDo: print new_converged_elites next to old_corresponding_elites
        # old_corresponding_elites -> right join by index
        # print("Mean Obj Improvement: {:.1f}".format(mean_obj_improvement))
        old_elite_obj = old_elites['objective']

        # Print Anytime Metrics
        print(f"\n\nObj Archive Size (before)     : {obj_t0}")
        print(f"Obj Archive Size  (after)     : {obj_t1}")
        print(f"Obj QD (per elite)            : {obj_qd_per_elite}")
        print(f"Obj QD (per bin)              : {obj_qd_per_bin}\n")

        print(f"Percentage New Obj Bins       : {percentage_new_obj_bins:.1f}%")
        print(f"Percentage Obj Improvements   : {percentage_obj_improvements:.1f}%")
        print(f"Percentage New Obj Elites     : {percentage_new_obj_elites:.1f}%")
        print(f"Total New Obj Bins            : {total_new_obj_bins}")
        print(f"Total Improvements            : {total_obj_improvements}")
        print(f"Total New Obj Elites          : {total_new_obj_elites}")
        print(f"Iteration New Obj Bins        : {n_new_obj_bins}")
        print(f"Iteration Improvements        : {n_new_obj_improvements}")
        print(f"Iteration New Obj Elites      : {n_new_obj_elites}\n")

        print(f"Percentage Convergence Errors : {percentage_convergence_errors:.1f}%")
        print(f"Total Convergence Errors      : {total_convergence_errors}")
        print(f"Iteration Convergence Errors  : {convergence_errors}\n")

        print(f"Percentage New Acq Bins       : {percentage_new_acq_bins:.1f}%")
        print(f"Percentage Acq Improvements   : {percentage_acq_improvements:.1f}%")
        print(f"Percentage New Acq Elites     : {percentage_new_acq_elites:.1f}%")
        print(f"Total New Acq Bins            : {total_new_acq_bins}")
        print(f"Total Improvements            : {total_acq_improvements}")
        print(f"Total New Acq Elites          : {total_new_acq_elites}")
        print(f"Iteration New Acq Bins        : {n_new_acq_bins}")
        print(f"Iteration Improvements        : {n_new_acq_improvements}")
        print(f"Iteration New Acq Elites      : {n_new_acq_elites}\n")

        print(f"Acq Archive Size (before)    : {acq_t0}")
        print(f"Acq Archive Size  (after)    : {acq_t1}")
        print(f"New Acq Elites               : {n_new_acq_elites}\n")

        print(f"Acq QD (per bin)             : {acq_qd_per_bin}")
        print(f"New Acq Bins                 : {n_new_acq_bins}")
        print(f"Mean Acq QD                  : {acq_qd_per_elite}")
        print(f"Remaining ACQ Precise Evals  : {current_eval_budget}\n\n")


        # Store Anytime Metrics in Pandas Dataframe
        anytime_data = [iteration, obj_qd_per_elite, obj_qd_per_bin,
                                   acq_qd_per_elite, acq_qd_per_bin,
                                   percentage_new_obj_elites, total_new_obj_elites, n_new_obj_elites,
                                   percentage_new_acq_elites, total_new_acq_elites, n_new_acq_elites,
                                   percentage_obj_improvements, total_obj_improvements, n_new_obj_elites,
                                   percentage_acq_improvements, total_acq_improvements, n_new_acq_elites,
                                   percentage_new_obj_bins, total_new_obj_bins, n_new_obj_bins,
                                   percentage_new_acq_bins, total_new_acq_bins, n_new_acq_bins,
                                   percentage_convergence_errors, total_convergence_errors, convergence_errors,]
        anytime_metrics.loc[len(anytime_metrics)] = anytime_data

        if iteration % 20 == 0:
            # Save Anytime Metrics to CSV. If no csv is saved create new csv, else append to existing csv
            try:
                anytime_metrics.to_csv(f'{self.initial_seed}_{self.domain}_acq_loop_anytime_metrics.csv', mode='a', header=False, index=False)
            except:
                anytime_metrics.to_csv(f'{self.initial_seed}_{self.domain}_acq_loop_anytime_metrics.csv', index=False)
            # Reset to empty dataframe
            anytime_metrics = pandas.DataFrame(columns= anytime_columns)
            
        iteration += 1



    if iteration % 20 == 0:
        gc.collect()

    return


def run_vanilla_sail(self: SailRun):
    return "Work to be done"


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


def run_random_sail(self: SailRun):
    """
    Extra evaluations are given if eval_pred_flag is True (see sail.py)
        if not eval_pred_flag, extra_evals = 0
    """

    obj_archive = self.obj_archive


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

        sol_array = np.vstack((sol_array, converged_samples)) # dtype=float64
        obj_array = np.vstack((obj_array, obj_batch.reshape(-1,1))) # dtype=float64

        eval_budget -= BATCH_SIZE

        print("\n\nAirfoil Convergence Errors: " + str(convergence_errors))
        print("Remaining Random Search Obj Evals: " + str(eval_budget) + "\n\n")

    return


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

    anytime_metrics = pandas.DataFrame(columns= ['Iteration',   'Mean Obj QD Score', 'Mean Obj QD Score per Bin', 'Mean pred QD Score', 'Mean pred QD Score per Bin', 'Percentage Improvements', 'Total Obj Improvements', 'New Obj Improvements','Percentage New Obj Bins', 'Total New Obj Bins', 'New Obj Bins','Percentage Convergence Errors', 'Total Convergence Errors','New Convergence Errors',])
    total_eval_budget = MAX_PRED_VERIFICATION
    total_pred_eval_budget = PRED_N_EVALS
    current_eval_budget = total_eval_budget
    current_pred_eval_budget = total_pred_eval_budget
    iter_evals = MAX_PRED_VERIFICATION//(PREDICTION_VERIFICATIONS)

    while(current_eval_budget >= iter_evals):

        print("Prediction Verification Loop")
        print("Iter Evals: " + str(iter_evals))
        print("Eval Budget: " + str(current_eval_budget))

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
        print(f"\n\nObj Archive Size (before)     : {obj_t0}")
        print(f"Obj Archive Size  (after)     : {obj_t1}")
        print(f"Obj QD (per elite)            : {obj_qd_per_elite}")
        print(f"Obj QD (per bin)              : {obj_qd_per_bin}\n")

        print(f"Percentage New Obj Bins       : {percentage_new_obj_bins:.1f}%")
        print(f"Percentage Obj Improvements   : {percentage_obj_improvements:.1f}%")
        print(f"Percentage New Obj Elites     : {percentage_new_obj_elites:.1f}%")
        print(f"Total New Obj Bins            : {total_new_obj_bins}")
        print(f"Total Improvements            : {total_obj_improvements}")
        print(f"Total New Obj Elites          : {total_new_obj_elites}")
        print(f"Iteration New Obj Bins        : {n_new_obj_bins}")
        print(f"Iteration Improvements        : {n_new_obj_improvements}")
        print(f"Iteration New Obj Elites      : {n_new_obj_elites}\n")

        print(f"Percentage Convergence Errors : {percentage_convergence_errors:.1f}%")
        print(f"Total Convergence Errors      : {total_convergence_errors}")
        print(f"Iteration Convergence Errors  : {convergence_errors}\n")

        print(f"Percentage New Pred Bins       : {percentage_new_pred_bins:.1f}%")
        print(f"Percentage Pred Improvements   : {percentage_pred_improvements:.1f}%")
        print(f"Percentage New Pred Elites     : {percentage_new_pred_elites:.1f}%")
        print(f"Total New Pred Bins            : {total_new_pred_bins}")
        print(f"Total Improvements            : {total_pred_improvements}")
        print(f"Total New Pred Elites          : {total_new_pred_elites}")
        print(f"Iteration New Pred Bins        : {n_new_pred_bins}")
        print(f"Iteration Improvements        : {n_new_pred_improvements}")
        print(f"Iteration New Pred Elites      : {n_new_pred_elites}\n")

        print(f"Pred Archive Size (before)    : {pred_t0}")
        print(f"Pred Archive Size  (after)    : {pred_t1}")
        print(f"New Pred Elites               : {n_new_pred_elites}\n")

        print(f"Pred QD (per bin)             : {pred_qd_per_bin}")
        print(f"New Pred Bins                 : {n_new_pred_bins}")
        print(f"Mean Pred QD                  : {pred_qd_per_elite}")
        print(f"Remaining ACQ Precise Evals  : {current_eval_budget}\n\n")


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

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

        solution_batch = create_sobol_samples(order=INIT_N_EVALS, dim=len(SOL_VALUE_RANGE), seed=self.current_seed+5)
        solution_batch = solution_batch.T
        measures_batch = solution_batch[:, 1:3]
        scale_samples(solution_batch) # sobol samples are between [0;1]
        eval_xfoil_loop(self, solution_batch=solution_batch, measures_batch=measures_batch, acq_flag=True) # fill obj archive inside eval_xfoil_loop() & update+render acq_archive
        print("\n[...] Terminate init_archive()\n")

    
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
        
        if candidate_sol.shape[0] == 0:
            return
        
        if candidate_obj is not None and (acq_flag or pred_flag):
            raise ValueError("update_archive: candidate_obj != None and acq_flag or pred_flag")
        
        if evaluate_prediction_archive:
            self.evaluated_predictions_archive.add(candidate_sol, candidate_obj, candidate_bhv)
            return
        if acq_flag:
            candidate_acq = acq_ucb(genomes=candidate_sol, gp_model=self.gp_model)
            self.acq_archive.add(candidate_sol, candidate_acq, candidate_bhv)
        if pred_flag:
            candidate_pred = predict_objective(genomes=candidate_sol, gp_model=self.gp_model)
            self.pred_archive.add(candidate_sol, candidate_pred, candidate_bhv)
    

    def define_archives(self, seed):

        obj_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = 1.0
        )

        acq_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = 1.0
        )

        pred_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = 1.0
        )

        # Used for visualizing new elites (improved + new bin discoveries)
        new_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = 1.0
        )

        # Used for evaluating quality of results
        evaluated_predictions_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = 1.0
        )

        # Used for visualizing prediction errors
        prediction_error_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = 0
        )

        return obj_archive, acq_archive, pred_archive, new_archive, evaluated_predictions_archive, prediction_error_archive


def run_vanilla_sail(self: SailRun):
    return "Work to be done"


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



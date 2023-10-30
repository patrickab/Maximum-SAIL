###### Import Foreign Scripts ######
from chaospy import create_sobol_samples
from ribs.archives import GridArchive
from gp.predict_objective import predict_objective
import gc
import numpy as np
import logging
import os

import subprocess
import PIL
from utils.anytime_archive_visualizer import archive_visualizer

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

from utils.utils import eval_xfoil_loop
from utils.anytime_archive_visualizer import anytime_archive_visualizer
from utils.pprint_nd import pprint

from xfoil.generate_airfoils import generate_parsec_coordinates
from xfoil.simulate_airfoils import xfoil

from acq_functions.acq_ucb import acq_ucb
from acq_functions.acq_mes import acq_mes
from gp.fit_gp_model import fit_gp_model
from map_elites import map_elites

class SailRun:

    def __init__(self, initial_seed, acq_ucb_flag=False, acq_mes_flag=False, sail_vanilla_flag=False, sail_custom_flag=False, sail_random_flag=False, pred_verific_flag=False, greedy_flag=False, explore_flag=False, hybrid_flag=False):

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
            self.domain = self.domain + "_verification"
        if greedy_flag and explore_flag:
            raise ValueError("Greedy and Explore Flags cannot both be True")
        if pred_verific_flag and not (greedy_flag or explore_flag or hybrid_flag):
            raise ValueError("Prediction Verification Flag requires Greedy or Explore Flag to be True")

        if acq_ucb_flag:
            self.acq_ucb_flag = True
            self.acq_mes_flag = False
            self.acq_function = acq_ucb
            self.domain = self.domain + "_ucb"
        if acq_mes_flag:
            self.acq_mes_flag = True
            self.acq_ucb_flag = False
            self.acq_function = acq_mes
            self.domain = self.domain + "_mes"

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

        if new_solutions.shape[0] == 0:
            return

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

        vmax = 5.0
        if self.acq_mes_flag and acq_flag:
            vmax = 0.6

        anytime_archive_visualizer(self, archive=archive, obj_flag=obj_flag, acq_flag=acq_flag, pred_flag=pred_flag, new_flag=new_flag, vmax=vmax)
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

        if candidate_sol.shape[0] == 0:
            return
        
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
        
        if candidate_obj is not None and (acq_flag or pred_flag):
            raise ValueError("update_archive: candidate_obj != None and acq_flag or pred_flag")
        
        if evaluate_prediction_archive:
            self.evaluated_predictions_archive.add(candidate_sol, candidate_obj, candidate_bhv)
            return
        if acq_flag:
            candidate_acq = self.acq_function(genomes=candidate_sol, gp_model=self.gp_model)
            self.acq_archive.add(candidate_sol, candidate_acq, candidate_bhv)
        if pred_flag:
            candidate_pred = predict_objective(genomes=candidate_sol, gp_model=self.gp_model)
            self.pred_archive.add(candidate_sol, candidate_pred, candidate_bhv)
    

    def define_archives(self, seed):

        # -log(x)=3 is equivalent to lift/drag ratio of 20 (in experimental set up to-be-defined as minimum for airfoil to be considered high quality)
        # eg a boeing 747 or Airbus A380 have a lift/drag ratio of 17-20 (https://en.wikipedia.org/wiki/Lift-to-drag_ratio)
        
        # therefore, in order to produce qualitative results, it makes sense to set a minimum threshold
        # in future work (for generalization), this threshold could be set as a hyperparameter or class attribute
        # https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/lift-to-drag-ratio/

        if self.acq_function == acq_ucb:
            min_acq_threshhold = 3.0
            min_obj_threshhold = 3.0
            min_pred_threshhold = 3.0
        if self.acq_function == acq_mes:
            min_acq_threshhold = 0.0
            min_obj_threshhold = 3.0
            min_pred_threshhold = 3.0

        obj_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = min_obj_threshhold
        )

        acq_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = min_acq_threshhold 
        )

        pred_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = min_pred_threshhold
        )

        # Used for visualizing new elites (improved + new bin discoveries)
        new_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = min_obj_threshhold
        )

        # Used for evaluating quality of results
        evaluated_predictions_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = -1000 # low perfoming predictions shall be stored under any circumstances
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
    


def scale_samples(samples, boundaries=SOL_VALUE_RANGE):
    """Scales Samples to boundaries"""

    # ToDo: vectorize
    for i in range(len(samples)):
        for j in range(len(samples[i])):
            lower_bound, upper_bound = boundaries[j]
            samples[i][j] = samples[i][j] *(upper_bound - lower_bound) + lower_bound

    return samples




def store_final_data(self: SailRun):

    acq_vmax = 5.0 if self.acq_ucb_flag else 1.0

    archive_visualizer(self=self, archive=self.obj_archive, prefix="obj", name="Objective Archive", min_val=1.0, max_val=5)
    archive_visualizer(self=self, archive=self.acq_archive, prefix="acq", name="Acquisition Archive", min_val=1.0, max_val=acq_vmax)
    archive_visualizer(self=self, archive=self.pred_archive, prefix="pred", name="Prediction Archive (unevaluated)", min_val=1.0, max_val=5)
    archive_visualizer(self=self, archive=self.evaluated_predictions_archive, prefix="evaluted_pred", name="Prediction Archive (evaluated)", min_val=1.0, max_val=5)
    archive_visualizer(self=self, archive=self.prediction_error_archive, prefix="error", name="Prediction Error Archive (percentual)", min_val=0, max_val=0.1)

    initial_seed = self.initial_seed
    domain = self.domain

    img_filenames = [f"imgs/{domain}/{initial_seed}/final_{initial_seed}_{domain}_{prefix}_heatmap.png" for prefix in ["obj", "acq", "pred", "evaluted_pred", "error"]]
    imgs = [PIL.Image.open(img) for img in img_filenames]
    imgs_comb = np.hstack([img for img in imgs])
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    imgs_comb.save(f'imgs/final_heatmaps_{initial_seed}_{domain}.png')

    subprocess.run(f"rm imgs/{domain}/{initial_seed}/*.png", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    obj_dataframe = self.obj_archive.as_pandas(include_solutions=True)
    obj_dataframe.to_csv(f"{initial_seed}_{domain}_obj_archive.csv", index=False)

    acq_dataframe = self.acq_archive.as_pandas(include_solutions=True)
    acq_dataframe.to_csv(f"{initial_seed}_{domain}_acq_archive.csv", index=False)

    pred_dataframe = self.pred_archive.as_pandas(include_solutions=True)
    pred_dataframe.to_csv(f"{initial_seed}_{domain}_pred_archive.csv", index=False)
    
    # This archive contains only converged predictions & their true objective values
    evaluated_pred_dataframe = self.evaluated_predictions_archive.as_pandas(include_solutions=True)
    evaluated_pred_dataframe.to_csv(f"{initial_seed}_{domain}_evaluated_pred_archive.csv", index=False)

    error_dataframe = self.prediction_error_archive.as_pandas(include_solutions=True)
    error_dataframe.to_csv(f"{initial_seed}_{domain}_error_archive.csv", index=False)


def evaluate_prediction_archive(self: SailRun):

    """
    Evaluate the predictions of the prediction archive.
    This is done to determine the quality of results.
    """

    print("Evaluate Prediction Archive")

    # Extract all elites from the prediction archive - (sorted by objective for nice visual effect during evaluation)
    unevaluated_prediction_elites = sorted(self.pred_archive, key=lambda x: x.objective, reverse=True)[:self.pred_archive.stats.num_elites]
    unevaluated_prediction_elites = np.array([(elite.solution, elite.index, elite.objective, elite.measures) for elite in unevaluated_prediction_elites], dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])
    unevaluated_prediction_solutions = unevaluated_prediction_elites['solution']
    unevaluated_prediction_measures = np.vstack(unevaluated_prediction_elites['behavior'])
    eval_xfoil_loop(self, solution_batch=unevaluated_prediction_solutions, measures_batch=unevaluated_prediction_measures, evaluate_prediction_archive=True, candidate_targetvalues=unevaluated_prediction_elites['objective'])

    # Extract all elites from the evaluated predictions archive - (sorted by index for comparison)
    evaluated_prediction_elites = sorted(self.evaluated_predictions_archive, key=lambda x: x.index)[:self.evaluated_predictions_archive.stats.num_elites]
    evaluated_prediction_elites = np.array([(elite.solution, elite.index, elite.objective, elite.measures) for elite in evaluated_prediction_elites], dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])
    unevaluated_prediction_elites = unevaluated_prediction_elites[np.argsort(unevaluated_prediction_elites['index'])]

    # Calculate mask for converged prediction elites
    is_converged_prediction_elite = np.isin(unevaluated_prediction_elites['index'], evaluated_prediction_elites['index'])

    # Extract converged prediction elites
    converged_unevaluated_prediction_elites = unevaluated_prediction_elites[is_converged_prediction_elite]
    converged_evaluated_prediction_elites = evaluated_prediction_elites

    prediction_error = converged_unevaluated_prediction_elites['objective'] - converged_evaluated_prediction_elites['objective']
    percentual_error = np.abs(prediction_error)/converged_evaluated_prediction_elites['objective']
    mpe_error = np.mean(percentual_error)
    mae_error = np.mean(np.abs(prediction_error))
    mse_error = np.mean(np.square(prediction_error))

    qd_obj = sum(self.obj_archive.as_pandas(include_solutions=True)['objective'].values)
    qd_pred = sum(self.pred_archive.as_pandas(include_solutions=True)['objective'].values)
    qd_pred_verified = sum(self.evaluated_predictions_archive.as_pandas(include_solutions=True)['objective'].values)
    n_bins = np.prod(self.obj_archive.dims)

    obj_qd_per_bin = round(qd_obj/n_bins, 1)
    pred_qd_per_bin = round(qd_pred/n_bins, 1)
    pred_verified_qd_per_bin = round(qd_pred_verified/n_bins, 1)
    obj_qd_per_elite = round(qd_obj/self.obj_archive.stats.num_elites, 1)
    pred_qd_per_elite = round(qd_pred/self.acq_archive.stats.num_elites, 1)
    pred_verified_qd_per_elite = round(qd_pred_verified/self.evaluated_predictions_archive.stats.num_elites, 1)

    self.evaluated_predictions_archive.add(np.vstack(converged_evaluated_prediction_elites['solution']), converged_evaluated_prediction_elites['objective'], np.vstack(converged_evaluated_prediction_elites['behavior']))
    self.prediction_error_archive.add(np.vstack(converged_unevaluated_prediction_elites['solution']), percentual_error, np.vstack(converged_unevaluated_prediction_elites['behavior']))

    percentual_errors_greater_than_005 = np.sum(np.abs(prediction_error)/converged_evaluated_prediction_elites['objective'] > 0.05)
    id_string = f"Initial Seed: {self.initial_seed}  Domain: {self.domain}\n"
    qd_string = f"Obj QD (per bin): {obj_qd_per_bin}\nPred QD (per bin / unverified): {pred_qd_per_bin}\nPred QD (per bin / verified): {pred_verified_qd_per_bin}\nObj QD (per elite): {obj_qd_per_elite}\nPred QD (per elite / unverified): {pred_qd_per_elite}\nPred QD (per elite / verified): {pred_verified_qd_per_elite}\n"
    error_str = f"MAE Error: {mae_error}\nMSE Error: {mse_error}\nMPE Error: {mse_error}\nPercentual Errors greater than 5%:  {percentual_errors_greater_than_005}\nPrediction Errors: \n{np.array2string(prediction_error)}\nPercentual Errors: \n{np.array2string(percentual_error)}\n"
    print("Percentual Errors Greater than 5%: ", percentual_errors_greater_than_005, "\n\n")
    
    with open("stats_log", "a") as file: 
        file.write(id_string)
        file.write(qd_string)
        file.write(error_str)

    true_objective = np.vstack(converged_evaluated_prediction_elites['objective'])
    predicted_objective = np.vstack(converged_unevaluated_prediction_elites['objective'])
    pprint(predicted_objective, true_objective, percentual_error)
    print("\nMAE Error: ", mae_error, "\n", "MSE Error: ", mse_error, "\n", "Mean Percentual Error: ", mpe_error, "\n")

    os.makedirs("csv") if not os.path.exists("csv") else None
    subprocess.run("mv *.csv csv", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    gc.collect()
    return
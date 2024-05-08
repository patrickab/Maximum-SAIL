"""
This module serves as a database class for the SAIL algorithm.
All data can easily be stored to and accessed from this class.
"""

from chaospy import create_sobol_samples
from ribs.archives import GridArchive
import numpy as np
import logging
import os
import subprocess
import PIL

#### Configurable Variables
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), 'config', 'config.ini'))
TEST_RUNS = config.TEST_RUNS
BATCH_SIZE = config.BATCH_SIZE
SOL_DIMENSION = config.SOL_DIMENSION
OBJ_DIMENSION = config.OBJ_DIMENSION
BHV_DIMENSION = config.BHV_DIMENSION
BHV_VALUE_RANGE = config.BHV_VALUE_RANGE
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SIGMA_MUTANTS = config.SIGMA_MUTANTS
MUTANT_CELLRANGE = config.MUTANT_CELLRANGE
OBJ_BHV_NUMBER_BINS = config.OBJ_BHV_NUMBER_BINS
ACQ_BHV_NUMBER_BINS = config.ACQ_BHV_NUMBER_BINS
MES_GRID_SIZE = config.MES_GRID_SIZE

#### Import Custom Scripts
from utils.anytime_archive_visualizer import anytime_archive_visualizer, archive_visualizer
from utils.pprint_nd import pprint
from acq_functions.acq_ucb import acq_ucb
from acq_functions.acq_mes import acq_mes
from gp.predict_objective import predict_objective
from gp.fit_gp_model import fit_gp_model
from map_elites import map_elites
import numpy as np

MAX_RENDER_THRESHHOLD = 5.5


class SailRun:

    def __init__(self, initial_seed, acq_ucb_flag=False, acq_mes_flag=False, sail_vanilla_flag=False, sail_custom_flag=False, botorch_flag=False, sail_random_flag=False, pred_verific_flag=False, mes_init=False):

        """
        Initialize a SAIL Run.

        Parameters

            initial_seed : used for seeding

            acq_ucb_flag : boolean flag for running SAIL with UCB acquisition function
            acq_mes_flag : boolean flag for running SAIL with MES acquisition function

            mes_init    : boolean flag for performing MES initialization before UCB loop

            sail_vanilla_flag : boolean flag for running SAIL with vanilla behavior
            sail_custom_flag  : boolean flag for running SAIL with custom behavior 
            sail_random_flag  : boolean flag for running SAIL with uniform random solutions

            pred_verific_flag : boolean flag for using prediction verification loop before returning prediction archive
            """

        # read into logger & use properly for DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialize SAIL Run")

        self.obj_current_iteration = 1
        self.new_current_iteration = 1
        self.acq_current_iteration = 1
        self.pred_current_iteration = 1
        self.map_current_iteration = 1

        if sail_custom_flag:
            self.domain = "custom"
        if sail_random_flag:
            self.domain = "randomsearch"
        if sail_vanilla_flag:
            self.domain = "vanilla"
        if botorch_flag:
            self.domain = "botorch_acqf"
        if pred_verific_flag:
            self.domain = self.domain + "_verification"
        if mes_init:
            self.domain = self.domain + "_mes_init"
        if acq_ucb_flag:
            self.domain = self.domain + "_ucb"
        if acq_mes_flag:
            self.domain = self.domain + "_mes"

        # stores new solutions from reevaluate_archive()
        self.new_sol = np.empty((0, SOL_DIMENSION))
        self.new_obj = np.empty((0, OBJ_DIMENSION))
        self.new_bhv = np.empty((0, BHV_DIMENSION))
        self.mes_elites = np.empty((0, SOL_DIMENSION))

        self.initial_seed = initial_seed
        self.current_seed = initial_seed

        self.convergence_errors = 0

        self.mes_init = mes_init

        self.custom_flag = sail_custom_flag
        self.vanilla_flag = sail_vanilla_flag
        self.random_flag = sail_random_flag

        self.pred_verific_flag = pred_verific_flag

        self.sol_array = np.empty((0, SOL_DIMENSION))
        self.obj_array = np.empty((0, OBJ_DIMENSION))

        self.acq_function = acq_mes
        self.acq_mes_flag = acq_mes_flag
        self.acq_ucb_flag = acq_ucb_flag

        self.anytime_metric_kwargs = None

        self.obj_archive, self.acq_archive, self.pred_archive, self.new_archive, self.evaluated_predictions_archive, self.prediction_error_archive =\
            self.define_archives(initial_seed)

        print("\n\n\nInitialize SAIL Run")
        print(f"Domain: {self.domain}")
        print(f"Initial Seed: {self.initial_seed}")    


    def update_gp_data(self, new_solutions, new_objectives):

        if new_solutions.shape[0] == 0:
            return
        
        print(f"Update GP Data [...]\n")
        n_new = new_solutions.shape[0]
        n_old = self.sol_array.shape[0]
        n_expected = n_old + n_new 
        new_solutions = np.vstack(new_solutions) if new_solutions.shape[0] != 11 else new_solutions
        new_objectives = np.vstack(new_objectives) if new_solutions.shape[0] != 0 else new_objectives
        self.sol_array = np.vstack((self.sol_array, new_solutions))
        self.obj_array = np.vstack((self.obj_array, new_objectives))
        n_resulted = self.sol_array.shape[0]

        if n_resulted != n_expected:
            raise ValueError("GP Data Update Error")


    def update_gp_model(self, new_solutions=None, new_objectives=None):

        self.gp_model = fit_gp_model(self.sol_array, self.obj_array)
        return
    

    def update_cellgrids(self):

        self.bhv_cellbounds, self.bhv_sobol_cellgrids, self.mes_sobol_cellgrid = mes_sobol_cellgrids(self, mutant_cellrange=-0.0001)
        return


    def update_mutant_cellgrids(self, mutant_cellrange=MUTANT_CELLRANGE):

        self.bhv_cellbounds_mutants, self.bhv_sobol_cellgrids_mutants, self.mes_sobol_cellgrid_mutants = mes_sobol_cellgrids(self, mutant_cellrange=mutant_cellrange)
        return


    def update_seed(self):

        self.current_seed += TEST_RUNS
        return self.current_seed


    def visualize_archive(self, archive, obj_flag=False, acq_flag=False, pred_flag=False, new_flag=False, map_flag=False):

        vmin = 0
        vmax = MAX_RENDER_THRESHHOLD

        if self.acq_mes_flag and (acq_flag or map_flag):
            vmin = 0.0
            vmax = 0.6

        # all visualisations of the acquisition archive represent the state of the archive, which candidate solutions are sampled from for objective evaluations

        anytime_archive_visualizer(self, archive=archive, obj_flag=obj_flag, acq_flag=acq_flag, pred_flag=pred_flag, new_flag=new_flag, map_flag=map_flag, vmin=vmin, vmax=vmax)
        if obj_flag:
            self.obj_current_iteration += 1
        if new_flag:
            self.new_current_iteration += 1
        if acq_flag:
            self.acq_current_iteration += 1
        if pred_flag:
            self.pred_current_iteration += 1
        if map_flag:
            self.map_current_iteration += 1

    def update_archive(self, candidate_sol=None, candidate_obj=None, candidate_bhv=None, obj_flag=False, acq_flag=False, pred_flag=False, evaluate_prediction_archive=False, niche_restricted_update=False, sigma_mutants=SIGMA_MUTANTS):
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
            self.n_new_obj_elites = self.new_archive.stats.num_elites

            return
        
        if candidate_obj is not None and (acq_flag or pred_flag):
            raise ValueError("update_archive: candidate_obj != None and acq_flag or pred_flag")
        
        if evaluate_prediction_archive:
            self.evaluated_predictions_archive.add(candidate_sol, candidate_obj, candidate_bhv)
            return

        if acq_flag:
            for i in range(0, candidate_sol.shape[0], BATCH_SIZE):

                if self.acq_ucb_flag:
                    i_candidate_sol = candidate_sol[i:i+BATCH_SIZE]
                    i_candidate_bhv = candidate_bhv[i:i+BATCH_SIZE]
                    i_candidate_acq = self.acq_function(self=self, genomes=i_candidate_sol)
                    self.acq_archive.add(i_candidate_sol, i_candidate_acq, i_candidate_bhv)

                elif self.acq_mes_flag:
                    i_candidate_sol = candidate_sol[i:i+BATCH_SIZE]
                    i_candidate_acq = self.acq_function(self=self, genomes=i_candidate_sol, niche_restricted_update=niche_restricted_update, sigma_mutants=sigma_mutants)

                    if i_candidate_sol.shape[0] != 0:
                        i_candidate_sol = self.mes_elites                 
                    else:
                        break

                    i_candidate_bhv = candidate_bhv[i:i+BATCH_SIZE]
                    self.acq_archive.add(i_candidate_sol, i_candidate_acq, i_candidate_bhv)


        if pred_flag:
            candidate_pred = predict_objective(self=self, genomes=candidate_sol)
            self.pred_archive.add(candidate_sol, candidate_pred, candidate_bhv)


    def define_archives(self, seed):

        obj_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=OBJ_BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            dtype=np.float64,
        )

        ACQ_BINS = ACQ_BHV_NUMBER_BINS if self.acq_mes_flag else OBJ_BHV_NUMBER_BINS
        acq_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=ACQ_BINS,
            ranges=BHV_VALUE_RANGE,
            dtype=np.float64,
        )

        pred_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=OBJ_BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            dtype=np.float64,
        )

        # Used for visualizing new elites (improved + new bin discoveries)
        new_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=OBJ_BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            dtype=np.float64,
        )

        # Used for evaluating quality of results
        evaluated_predictions_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=OBJ_BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            dtype=np.float64,
        )

        # Used for visualizing prediction errors
        prediction_error_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=OBJ_BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            dtype=np.float64
        )

        return obj_archive, acq_archive, pred_archive, new_archive, evaluated_predictions_archive, prediction_error_archive


def store_final_data(self: SailRun):

    max_acq_threshhold = MAX_RENDER_THRESHHOLD if self.acq_ucb_flag else 0.6

    min_obj_threshhold = 0
    min_pred_threshhold = 0
    min_acq_threshhold = 0

    archive_visualizer(self=self, archive=self.obj_archive, prefix="obj", name="Objective Archive", min_val=min_obj_threshhold, max_val=MAX_RENDER_THRESHHOLD)
    archive_visualizer(self=self, archive=self.acq_archive, prefix="acq", name="Acquisition Archive", min_val=min_acq_threshhold, max_val=max_acq_threshhold)
    archive_visualizer(self=self, archive=self.pred_archive, prefix="pred", name="Prediction Archive (unevaluated)", min_val=min_pred_threshhold, max_val=MAX_RENDER_THRESHHOLD)
    archive_visualizer(self=self, archive=self.evaluated_predictions_archive, prefix="evaluted_pred", name="Prediction Archive (evaluated)", min_val=min_pred_threshhold, max_val=MAX_RENDER_THRESHHOLD)
    archive_visualizer(self=self, archive=self.prediction_error_archive, prefix="error", name="Prediction Error Archive (percentual)", min_val=0, max_val=0.10)
    #                                                        render maximum of 10% error (for better visualization) - errors above 10% are stored in stats_log

    initial_seed = self.initial_seed
    domain = self.domain

    img_filenames = [f"imgs/{domain}/{initial_seed}/final_{initial_seed}_{domain}_{prefix}_heatmap.png" for prefix in ["obj", "pred", "evaluted_pred", "error"]]
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

    from xfoil.eval_xfoil_loop import eval_xfoil_loop

    print("Evaluate Prediction Archive")

    # Extract all elites from the prediction archive - (sorted by objective for nice visual effect during evaluation)
    unevaluated_prediction_elites = self.pred_archive.as_pandas(include_solutions=True).sort_values(by=['objective'], ascending=False)

    unevaluated_prediction_objectives = unevaluated_prediction_elites.objective_batch()
    unevaluated_prediction_solutions = unevaluated_prediction_elites.solution_batch()
    unevaluated_prediction_measures = unevaluated_prediction_elites.measures_batch()

    eval_xfoil_loop(self, solution_batch=unevaluated_prediction_solutions, measures_batch=unevaluated_prediction_measures, evaluate_prediction_archive=True, candidate_targetvalues=unevaluated_prediction_objectives)

    # Extract all elites from the evaluated predictions archive - (sorted by index for comparison)
    evaluated_prediction_elites = self.evaluated_predictions_archive.as_pandas(include_solutions=True)
    evaluated_prediction_elites = evaluated_prediction_elites.sort_values(by=['index'], ascending=True)
    unevaluated_prediction_elites = unevaluated_prediction_elites.sort_values(by=['index'], ascending=True)

    # Calculate mask for converged prediction elites
    evaluated_solution_indices = evaluated_prediction_elites['index']
    unevaluated_solution_indices = unevaluated_prediction_elites['index']
    is_converged_prediction_elite = np.isin(unevaluated_solution_indices, evaluated_solution_indices)

    # Drop all non-converged prediction elites
    unevaluated_predictions = unevaluated_prediction_elites[is_converged_prediction_elite]
    evaluated_predictions = evaluated_prediction_elites

    prediction_error = unevaluated_predictions.objective_batch() - evaluated_predictions.objective_batch()
    percentual_error = np.abs(prediction_error)/evaluated_predictions.objective_batch()
    mpe_error = np.mean(percentual_error)
    mae_error = np.mean(np.abs(prediction_error))
    mse_error = np.mean(np.square(prediction_error))

    qd_obj = sum(self.obj_archive.as_pandas(include_solutions=True)['objective'].values)
    qd_pred = sum(self.pred_archive.as_pandas(include_solutions=True)['objective'].values)
    qd_pred_verified = sum(self.evaluated_predictions_archive.as_pandas(include_solutions=True)['objective'].values)
    n_cells = np.prod(self.obj_archive.dims)

    obj_qd_per_bin = round(qd_obj/n_cells, 1)
    pred_qd_per_bin = round(qd_pred/n_cells, 1)
    pred_verified_qd_per_bin = round(qd_pred_verified/n_cells, 1)
    obj_qd_per_elite = round(qd_obj/self.obj_archive.stats.num_elites, 1)
    pred_qd_per_elite = round(qd_pred/self.acq_archive.stats.num_elites, 1)
    pred_verified_qd_per_elite = round(qd_pred_verified/self.evaluated_predictions_archive.stats.num_elites, 1)

    self.prediction_error_archive.add(evaluated_prediction_elites.solution_batch(), percentual_error, evaluated_prediction_elites.measures_batch())

    percentual_errors_greater_than_10 = np.sum(np.abs(prediction_error)/evaluated_prediction_elites.objective_batch() > 0.1)
    id_string = f"Initial Seed: {self.initial_seed}  Domain: {self.domain}\n"
    qd_string = f"Obj QD (per bin): {obj_qd_per_bin}\nPred QD (per bin / unverified): {pred_qd_per_bin}\nPred QD (per bin / verified): {pred_verified_qd_per_bin}\nObj QD (per elite): {obj_qd_per_elite}\nPred QD (per elite / unverified): {pred_qd_per_elite}\nPred QD (per elite / verified): {pred_verified_qd_per_elite}\n"
    obj_elites_stats = f"Highest Objective Value: {self.obj_archive.best_elite.objective}   Number of Objective Elites: {self.obj_archive.stats.num_elites}   Objective values higher than 5.00: {np.sum(self.obj_archive.as_pandas(include_solutions=True)['objective'].values > 5.00)}\n"
    verified_elites_stats = f"Highest Verified Obj Value: {self.evaluated_predictions_archive.best_elite.objective}   Number of Verified Elites: {self.evaluated_predictions_archive.stats.num_elites}   Verified objectives values higher than 5.00: {np.sum(self.acq_archive.as_pandas(include_solutions=True)['objective'].values > 5.00)}\n"
    error_str = f"MAE Error: {mae_error}\nMSE Error: {mse_error}\nMPE Error: {mse_error}\nPercentual Errors greater than 5%:  {percentual_errors_greater_than_10}\nPrediction Errors: \n{np.array2string(prediction_error)}\nPercentual Errors: \n{np.array2string(percentual_error)}\n"
    print("Percentual Errors Greater than 10%: ", percentual_errors_greater_than_10, "\n\n")
    
    with open("stats_log", "a") as file: 
        file.write(id_string)
        file.write(qd_string)
        file.write(obj_elites_stats)
        file.write(verified_elites_stats)
        file.write(error_str)

    true_obj = evaluated_predictions.objective_batch()
    pred_obj = unevaluated_predictions.objective_batch()
    pprint(true_obj, pred_obj, percentual_error)
    print("\nMAE Error: ", mae_error, "\n", "MSE Error: ", mse_error, "\n", "Mean Percentual Error: ", mpe_error, "\n")

    os.makedirs("csv") if not os.path.exists("csv") else None
    subprocess.run("mv *.csv csv", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return


def mes_sobol_cellgrids(self: SailRun, mutant_cellrange):

    """
    Creates a Sobol Cellgrid that can be used for all cells

        This function is called once before every MAP-Elites-Loop.
    
        Among non-measure dimensions, all sobol samples are equal.
        Therefore we need to draw only one sobol sample for all grids.

        Seperation of bhv_cellgrids and mes_cellgrids allows us to
        scale the behavior space independently from the solution space,
        which accelerates calculation, while reducing also reducing
        memory consumption significantly.

        Mes/Bhv Cellgrids are stored within the SailRunner class.
        Mes Cellgrid is constant across all bins.
        Bhv Cellgrid can be accessed by index.

        Therefore, we can rapidly assamble the final cellgrid
        for each sample within the MAP-Loop

    Mutant Cellrange:

        Defines the boundaries for the cellgrids.
        These boundaries are also used as boundaries
        for the mutants, that are created inside acq_mes.py

        For example:
            mutant_cellrange = 0.1 -> mutants may exceed cellbounds by 10%
            mutant_cellrange = 0.0 -> mutants are not allowed to exceed cellbounds
            mutant_cellrange = -0.01 -> mutants are not allowed to exceed cellbounds, but are allowed to be sampled from a smaller cell

        Setting mutant_cellrange to -0.01 avoids edge cases


    Returns:

        bhv_cellbounds : used for scaling gaussian mutants to cellbounds
        bhv_cellgrids  : used for assambling cellgrid (for individual genomes)
        mes_cellgrid   : used for assambling cellgrid (shared across all genomes)

    # visual representation : https://github.com/patrickab/Maximum-SAIL/blob/master/docs/MES%20Sobol%20Cellgrids.pdf

    """
    sobol_cellgrid = create_sobol_samples(order=MES_GRID_SIZE, dim=SOL_DIMENSION, seed=self.current_seed).T

    archive = self.acq_archive
    n_cells = np.prod(archive.dims)

    archive_indices = range(n_cells)
    idx = archive.int_to_grid_index(archive_indices)

    lower_bounds = np.array(SOL_VALUE_RANGE)[:, 0]
    upper_bounds = np.array(SOL_VALUE_RANGE)[:, 1]

    bhv_cellgrid = sobol_cellgrid[:, 1:3]
    mes_cellgrid = sobol_cellgrid * (upper_bounds - lower_bounds) + lower_bounds

    boundaries_0 = archive.boundaries[0]
    boundaries_1 = archive.boundaries[1]

    cell_range_0 = np.diff(boundaries_0)[0]
    cell_range_1 = np.diff(boundaries_1)[0]

    bhv_cellgrids = np.empty((n_cells, MES_GRID_SIZE, BHV_DIMENSION))
    bhv_cellbounds = np.empty((n_cells, BHV_DIMENSION, 2))

    for i in range(n_cells):

        measure_0_idx, measure_1_idx = idx[i]

        cell_bounds_0 = (boundaries_0[measure_0_idx] - cell_range_0 * mutant_cellrange,
                         boundaries_0[measure_0_idx+1] + cell_range_0 * mutant_cellrange)

        cell_bounds_1 = (boundaries_1[measure_1_idx] - cell_range_1 * mutant_cellrange,
                         boundaries_1[measure_1_idx+1] + cell_range_1 * mutant_cellrange)

        # Restrict cellbounds to solution space boundaries
        cell_bounds_0 = np.clip(cell_bounds_0, boundaries_0[0], boundaries_0[-1])
        cell_bounds_1 = np.clip(cell_bounds_1, boundaries_1[0], boundaries_1[-1])

        cell_bounds_i = np.array([cell_bounds_0, cell_bounds_1])
        bhv_cellbounds[i] = cell_bounds_i

        lower_bounds = cell_bounds_i[:, 0]
        upper_bounds = cell_bounds_i[:, 1]
        cell_bound_ranges = upper_bounds - lower_bounds

        bhv_cellgrid_i = bhv_cellgrid.copy()        
        bhv_cellgrid_i = bhv_cellgrid_i * cell_bound_ranges.T + lower_bounds   # scale sobol cellgrid to cellbounds
        bhv_cellgrids[i] = bhv_cellgrid_i                                      # insert bhv cellgrid into mes cellgrid

    return bhv_cellbounds, bhv_cellgrids, mes_cellgrid


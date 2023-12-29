###### Import Foreign Scripts ######
from ribs.archives import GridArchive
import gc
import numpy as np
import logging
import os
import subprocess
import PIL

### Configurable Variables ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), 'config', 'config.ini'))
TEST_RUNS = config.TEST_RUNS
BATCH_SIZE = config.BATCH_SIZE
INIT_N_EVALS = config.INIT_N_EVALS
SOL_DIMENSION = config.SOL_DIMENSION
OBJ_DIMENSION = config.OBJ_DIMENSION
BHV_DIMENSION = config.BHV_DIMENSION
MES_GRID_SIZE = config.MES_GRID_SIZE
BHV_VALUE_RANGE = config.BHV_VALUE_RANGE
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
MUTANT_CELLRANGE = config.MUTANT_CELLRANGE
OBJ_BHV_NUMBER_BINS = config.OBJ_BHV_NUMBER_BINS
ACQ_BHV_NUMBER_BINS = config.ACQ_BHV_NUMBER_BINS

###### Import Custom Scripts ######
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

    def __init__(self, initial_seed, acq_ucb_flag=False, acq_mes_flag=False, sail_vanilla_flag=False, sail_custom_flag=False, sail_random_flag=False, pred_verific_flag=False, greedy_flag=False, hybrid_flag=False, random_init=False, mes_init=False, ucb_init=False):

        """
        Initialize a SAIL Run.

        Parameters
    
            initial_seed : used for seeding - among each benchmark iteration, all algorithms are seeded with the same value. This value is incremented by the number of TEST_RUNS, to ensure a unique sequence of seeds for each benchmark iteration, identical across all benchmarked algorithms.
        
            acq_ucb_flag : boolean flag for running SAIL with UCB acquisition function
            acq_mes_flag : boolean flag for running SAIL with MES acquisition function
        
            greedy_flag: boolean flag for sampling only highest performing solutions from new elites archive (100% exploitation)
            hybrid_flag: boolean flag for sampling certain percentage of highest performing solutions, certain percentage of new bin solutions (xx.xx% exploitation, 100-xx.xx% exploration)
        
            random_init: boolean flag for initializing target archive with quasi-random sobol samples
            mes_init: boolean flag for initializing target archive with MES samples
        
            sail_vanilla_flag : boolean flag for running SAIL with vanilla behavior
                - before each MAP-Loop, initialize target archive with objective elites
            sail_custom_flag : boolean flag for running SAIL with custom behavior 
                - before each MAP-Loop, initialize target archive with objective elites & updated target elites
                - requires selection of: (greedy_flag or hybrid_flag) and (random_init or mes_init)
                - offers possibility of using pred_verific_flag
            sail_random_flag : boolean flag for running SAIL with uniform random solutions
            """

        # read into logger & use properly for DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initialize SAIL Run")

        self.obj_current_iteration = 1
        self.new_current_iteration = 1
        self.acq_current_iteration = 1
        self.pred_current_iteration = 1
        self.map_current_iteration = 1


        if hybrid_flag:
            self.domain = "hybrid"
        if greedy_flag:
            self.domain = "greedy"
        if sail_random_flag:
            self.domain = "random"
        if sail_vanilla_flag:
            self.domain = "vanilla"
        if pred_verific_flag:
            self.domain = self.domain + "_verification"
        if ucb_init:
            self.domain = self.domain + "_ucb_init"
        if mes_init:
            self.domain = self.domain + "_mes_init"
        if random_init:
            self.domain = self.domain + "_random_init"
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
        self.ucb_init = ucb_init
        self.random_init = random_init
        self.random_init_flag = random_init #todo: remove

        self.custom_flag = sail_custom_flag
        self.vanilla_flag = sail_vanilla_flag
        self.random_flag = sail_random_flag

        self.greedy_flag = greedy_flag
        self.hybrid_flag = hybrid_flag
        self.pred_verific_flag = pred_verific_flag

        self.sol_array = np.empty((0, SOL_DIMENSION))
        self.obj_array = np.empty((0, OBJ_DIMENSION))

        self.acq_function = acq_mes
        self.acq_mes_flag = acq_mes_flag
        self.acq_ucb_flag = acq_ucb_flag

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

    def update_archive(self, gp=None, candidate_sol=None, candidate_obj=None, candidate_bhv=None, obj_flag=False, acq_flag=False, pred_flag=False, eval_pred_archive=False, niche_restricted_update=False, acq_archive=None):
        """"
        Input:
            Option 1: Call with archive & archive flag
            Option 2: Call with candidate_sol, candidate_obj, candidate_bhv & archive flag
        """            

        if acq_archive is not None:
            self.acq_archive = acq_archive
            return

        if candidate_sol.shape[0] == 0:
            return
        
        target_function = self.acq_function if acq_flag else predict_objective
        target_archive = self.acq_archive if acq_flag else self.pred_archive
        ucb_flag = self.acq_ucb_flag
        mes_flag = self.acq_mes_flag

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
                
        if eval_pred_archive:
            self.evaluated_predictions_archive.add(candidate_sol, candidate_obj, candidate_bhv)
            return

        if self.acq_mes_flag and acq_flag:
            target_archive.add(solution_batch=candidate_sol, objective_batch=candidate_obj, measures_batch=candidate_bhv)
            return

        if self.acq_ucb_flag and acq_flag:

            for i in range(0, candidate_sol.shape[0], BATCH_SIZE):

                i_candidate_sol = candidate_sol[i:i+BATCH_SIZE]
                i_candidate_bhv = candidate_bhv[i:i+BATCH_SIZE]

                if ucb_flag:
                    i_candidate_acq = target_function(self=self, genomes=i_candidate_sol, gp_model=gp)
                elif mes_flag:
                    i_candidate_acq = target_function(self=self, genomes=i_candidate_sol, niche_restricted_update=niche_restricted_update, gp_model=gp)

                target_archive.add(i_candidate_sol, i_candidate_acq, i_candidate_bhv)

        if pred_flag:
            candidate_pred = predict_objective(self=self, genomes=candidate_sol)
            self.pred_archive.add(candidate_sol, candidate_pred, candidate_bhv)


    def define_archives(self, seed):

        # -log(x)=2.5 is equivalent to lift/drag ratio of 12.18
        # eg a boeing 747 or Airbus A380 have a lift/drag ratio of 17-20 (https://en.wikipedia.org/wiki/Lift-to-drag_ratio)
        # https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/lift-to-drag-ratio/

        # therefore, in order to produce qualitative results, it makes sense to set a minimum threshold

        obj_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=OBJ_BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
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

    from xfoil.eval_xfoil_loop import eval_xfoil_loop

    print("Evaluate Prediction Archive")

    # Extract all elites from the prediction archive - (sorted by objective for nice visual effect during evaluation)
    unevaluated_prediction_elites = self.pred_archive.as_pandas(include_solutions=True).sort_values(by=['objective'], ascending=False)

    unevaluated_prediction_objectives = unevaluated_prediction_elites.objective_batch()
    unevaluated_prediction_solutions = unevaluated_prediction_elites.solution_batch()
    unevaluated_prediction_measures = unevaluated_prediction_elites.measures_batch()
    eval_xfoil_loop(self, solution_batch=unevaluated_prediction_solutions, measures_batch=unevaluated_prediction_measures, eval_pred_archive=True, candidate_targetvalues=unevaluated_prediction_objectives)

    # Extract all elites from the evaluated predictions archive - (sorted by index for comparison)
    evaluated_prediction_elites = self.evaluated_predictions_archive.as_pandas(include_solutions=True)
    evaluated_prediction_elites = evaluated_prediction_elites.sort_values(by=['index'], ascending=True)
    unevaluated_prediction_elites = unevaluated_prediction_elites.sort_values(by=['index'], ascending=True)

    # Calculate mask for converged prediction elites
    evaluated_solution_batch = evaluated_prediction_elites.solution_batch()
    unevaluated_solution_batch = unevaluated_prediction_elites.solution_batch()
    is_converged_prediction_elite = np.isin(unevaluated_solution_batch, evaluated_solution_batch).all(1)

    # Extract converged prediction elites
    unevaluated_predictions = unevaluated_prediction_elites[np.isin(unevaluated_prediction_elites.solution_batch(), evaluated_prediction_elites.solution_batch()).all(1)]
    evaluated_predictions = evaluated_prediction_elites[np.isin(evaluated_prediction_elites.solution_batch(), unevaluated_predictions.solution_batch()).all(1)]

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

    true_objective = evaluated_prediction_elites.objective_batch()
    predicted_objective = unevaluated_prediction_elites.objective_batch()
    pprint(predicted_objective, true_objective, percentual_error)
    print("\nMAE Error: ", mae_error, "\n", "MSE Error: ", mse_error, "\n", "Mean Percentual Error: ", mpe_error, "\n")

    os.makedirs("csv") if not os.path.exists("csv") else None
    subprocess.run("mv *.csv csv", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    gc.collect()
    return
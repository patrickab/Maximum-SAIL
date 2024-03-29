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
ACQ_UCB_MIN_THRESHHOLD = config.ACQ_UCB_MIN_THRESHHOLD
ACQ_MES_MIN_THRESHHOLD = config.ACQ_MES_MIN_THRESHHOLD
INIT_N_ACQ_EVALS = config.INIT_N_ACQ_EVALS
INIT_N_SOBOL_ACQ = config.INIT_N_SOBOL_ACQ
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
ACQ_N_OBJ_EVALS = config.ACQ_N_OBJ_EVALS
ACQ_N_MAP_EVALS = config.ACQ_N_MAP_EVALS
PRED_N_OBJ_EVALS = config.PRED_N_OBJ_EVALS
PRED_N_MAP_EVALS = config.PRED_N_MAP_EVALS
SOL_DIMENSION = config.SOL_DIMENSION
INIT_N_EVALS = config.INIT_N_EVALS
BATCH_SIZE = config.BATCH_SIZE

n_obj_evals = INIT_N_EVALS + ACQ_N_OBJ_EVALS + PRED_N_OBJ_EVALS + INIT_N_ACQ_EVALS
CSV_BUFFERSIZE = (n_obj_evals/BATCH_SIZE) / 8

###### Import Custom Scripts ######

from xfoil.eval_xfoil_loop import eval_xfoil_loop
from utils.pprint_nd import pprint
from utils.anytime_metrics import initialize_anytime_metrics, calculate_anytime_metrics, store_anytime_metrics
from xfoil.generate_airfoils import generate_parsec_coordinates
from acq_functions.acq_mes import acq_mes
from acq_functions.acq_ucb import acq_ucb

from map_elites import map_elites
from sail_runner import SailRun
from chaospy import create_sobol_samples


def run_random_sail(self: SailRun):

    initialize_archive(self)

    ranges = np.array(SOL_VALUE_RANGE)

    def uniform_sample():
        uniform_sample = np.random.uniform(ranges[:, 0], ranges[:, 1], SOL_DIMENSION)
        return uniform_sample

    random_samples = np.array([uniform_sample() for i in range(n_obj_evals)])
    measures_batch = random_samples[:, 0:2]

    generate_parsec_coordinates(random_samples)
    obj_t0, obj_t1, n_new_obj_elites = eval_xfoil_loop(self, solution_batch=random_samples, measures_batch=measures_batch, acq_flag=True)

    for i in range(n_obj_evals//BATCH_SIZE):
        self.visualize_archive(self.acq_archive, acq_flag=True)
    
    self.update_gp_model()

    return


def run_vanilla_sail(self: SailRun):

    initialize_archive(self)

    anytime_metric_kwargs = initialize_anytime_metrics(self=self, acq_flag=True)
    current_eval_budget = anytime_metric_kwargs['current_eval_budget']

    while(current_eval_budget >= BATCH_SIZE):

        # Produce new acquisition elites
        target_t0 = self.acq_archive.stats.num_elites
        new_acq_elites, _, _ = map_elites(self, acq_flag=True)
        if new_acq_elites.stats.num_elites < BATCH_SIZE: ensure_n_new_elites(self=self, new_elite_archive=new_acq_elites, acq_flag=True)                  # Sample until enough new acquisition elites are found
        candidate_solutions_df = self.acq_archive.as_pandas(include_solutions=True).sample(n=BATCH_SIZE, random_state=self.initial_seed, replace=False)   # IMPORTANT ToDo: sobol sample in seed paper
        target_t1 = self.acq_archive.stats.num_elites

        self.visualize_archive(archive=self.acq_archive, acq_flag=True)

        improved_elites, new_bin_elites = prepare_sample_elites(self=self, new_elite_archive=new_acq_elites, old_elite_archive=self.obj_archive)  # Seperate new acq elites for calculating anytime metrics (not used for sampling)

        solution_batch = candidate_solutions_df.solution_batch()
        objective_batch = candidate_solutions_df.objective_batch()
        measures_batch = candidate_solutions_df.measures_batch()

        obj_t0, obj_t1, n_new_obj_elites = eval_xfoil_loop(self, solution_batch=solution_batch, measures_batch=measures_batch, acq_flag=True, candidate_targetvalues=objective_batch)     # Evaluate Acquisition Elites & Update Acq Archive under resulting GP Model

        # Calculate, Print & Store Anytime Metrics
        anytime_metric_kwargs = calculate_anytime_metrics(self=self, obj_t0=obj_t0, obj_t1=obj_t1, target_t0=target_t0, target_t1=target_t1, n_new_obj_elites=n_new_obj_elites, new_target_bin_elites=new_bin_elites, improved_target_elites=improved_elites, anytime_metric_kwargs=anytime_metric_kwargs, acq_flag=True)
        store_anytime_metrics(self, acq_flag=True, anytime_metric_kwargs=anytime_metric_kwargs)

        iteration = anytime_metric_kwargs['iteration']
        current_eval_budget = anytime_metric_kwargs['current_eval_budget']

        if iteration % 20 == 0:
            gc.collect()

    return


def run_custom_sail(self: SailRun, acq_loop=False, pred_loop=False):
    """
    Args:
        acq_loop:   If True, will enter acquisition loop
        pred_loop:  If True, will enter prediction verification loop
    """

    if not pred_loop:
        initialize_archive(self)

    CURIOSITY = 5 # For Hybrid Approach: 'CURIOSITY//BATCH_SIZE' new bin elites are to be sampled

    anytime_metric_kwargs = initialize_anytime_metrics(self=self, acq_flag=acq_loop, pred_flag=pred_loop)

    total_eval_budget = anytime_metric_kwargs['total_eval_budget']
    current_eval_budget = anytime_metric_kwargs['current_eval_budget']
    consumed_obj_evals = anytime_metric_kwargs['consumed_obj_evals']

    if acq_loop:
        i_obj_evals = BATCH_SIZE
        target_archive = self.acq_archive
    if pred_loop:
        i_obj_evals = PRED_N_OBJ_EVALS//PREDICTION_VERIFICATIONS
        target_archive = self.pred_archive

    while(current_eval_budget >= i_obj_evals):

        if consumed_obj_evals == total_eval_budget//3:
            CURIOSITY = 4
            if total_eval_budget == total_eval_budget//6:
                CURIOSITY = 3

        if (consumed_obj_evals % (total_eval_budget//10)) == 0 and consumed_obj_evals != 0:

            if self.acq_mes_flag:
                # preserve only highperforming elites
                target_elites = target_archive.as_pandas(include_solutions=True)
                update_elites = target_elites[target_elites.shape[0]//2]
                self.acq_archive.clear()
                self.update_archive(candidate_sol=update_elites.solution_batch(), candidate_bhv=update_elites.measures_batch(), acq_flag=True)

            # initialize acq archive with sobol samples
            n_sobol_samples = 1200
            solution_batch = create_sobol_samples(order=n_sobol_samples, dim=len(SOL_VALUE_RANGE), seed=self.current_seed+5)
            solution_batch = solution_batch.T
            solution_batch = scale_samples(solution_batch)
            measures_batch = solution_batch[:, 1:3]
            for i in range(0, n_sobol_samples, BATCH_SIZE):
                self.update_archive(candidate_sol=solution_batch[i:i+BATCH_SIZE], candidate_bhv=measures_batch[i:i+BATCH_SIZE], acq_flag=acq_loop, pred_flag=pred_loop)
                print(f"Sobol iteration: {i}")
            del solution_batch, measures_batch
            gc.collect()

        # Produce new acquisition elites
        target_t0 = target_archive.stats.num_elites
        new_target_elites, _, _ = map_elites(self, acq_flag=acq_loop, pred_flag=pred_loop)
        if new_target_elites.stats.num_elites < BATCH_SIZE: new_target_elites = ensure_n_new_elites(self=self, new_elite_archive=new_target_elites, acq_flag=acq_loop, pred_flag=pred_loop)   # Sample until enough new acquisition elites are found
        improved_elites, new_bin_elites = prepare_sample_elites(self=self, new_elite_archive=new_target_elites, old_elite_archive=self.obj_archive, pred_flag=pred_loop)                      # Split new_target_elites into improved elites & new bin elites, then (if self.acq_ucb_flag or pred_flag) calculate objective improvement (else) objective_improvement = objective
        candidate_solutions_df = select_samples(self, improved_elites=improved_elites, new_bin_elites=new_bin_elites, acq_flag=acq_loop, pred_flag=pred_loop, curiosity=CURIOSITY)            # Select samples based on exploration behavior defined in the class constructor
        target_t1 = target_archive.stats.num_elites


        # visualize resulting archives (multiple times during prediction verification to ensure videos of equal length)
        iterations = i_obj_evals//BATCH_SIZE
        while iterations > 0:
            self.visualize_archive(archive=self.acq_archive, acq_flag=True)
            if pred_loop:
                self.visualize_archive(archive=self.pred_archive, pred_flag=True)
            iterations -= 1


        solution_batch = candidate_solutions_df.solution_batch()
        objective_batch = candidate_solutions_df.objective_batch()
        measures_batch = candidate_solutions_df.measures_batch()

        if np.any(np.isin(solution_batch, self.sol_array).all(1)):
            raise ValueError("Duplicate Solution Error: New Solutions already exist in GP Data")

        obj_t0, obj_t1, n_new_obj_elites = eval_xfoil_loop(self, solution_batch=solution_batch, measures_batch=measures_batch, candidate_targetvalues=objective_batch, acq_flag=acq_loop, pred_flag=pred_loop)       # Evaluate Acquisition Elites & Update Acq Archive under resulting GP Model

        # Calculate, Print & Store Anytime Metrics
        anytime_metric_kwargs = calculate_anytime_metrics(self=self, obj_t0=obj_t0, obj_t1=obj_t1, target_t0=target_t0, target_t1=target_t1, n_new_obj_elites=n_new_obj_elites, new_target_bin_elites=new_bin_elites, improved_target_elites=improved_elites, anytime_metric_kwargs=anytime_metric_kwargs, acq_flag=acq_loop, pred_flag=pred_loop)
        store_anytime_metrics(self=self, acq_flag=acq_loop, pred_flag=pred_loop, anytime_metric_kwargs=anytime_metric_kwargs)

        iteration = anytime_metric_kwargs['iteration']
        current_eval_budget = anytime_metric_kwargs['current_eval_budget']

        if iteration % 20 == 0:
            gc.collect()

    return


def ensure_n_new_elites(self: SailRun, new_elite_archive, acq_flag=False, pred_flag=False):

    """
    - Ensures that the appropiate number of candidate solutions is availible for evaluation
    - After 3 extra MAP-Loops, the function returns the best elites found so far to avoid infinite loops

    Inputs:
        New Elite Archive
        Acq Flag XOR Pred Flag
    """

    target = "Acq" if acq_flag else "Pred"

    if acq_flag:
        n_samples = BATCH_SIZE
    if pred_flag and self.pred_verific_flag:
        n_samples = PRED_N_OBJ_EVALS//PREDICTION_VERIFICATIONS
    if pred_flag and not self.pred_verific_flag:
        raise ValueError("Maximize Improvement: Prediction Flag is True, but Prediction Verification Flag is False")

    # Re-enter MAP-Elites (acq/obj) up to 2 times in order to produce new elites 
    iteration = 0
    while new_elite_archive.stats.num_elites < n_samples and iteration <= 3:
        iteration += 1

        print(f'\n\nNot enough {target} Improvements: Re-entering {target}')
        print(f'New {target} Elites (before): {new_elite_archive.stats.num_elites}')
        new_elite_archive, _, _ = map_elites(self, new_elite_archive=new_elite_archive, acq_flag=acq_flag, pred_flag=pred_flag, re_enter_flag=True)
        print(f'New {target} Elites (after):  {new_elite_archive.stats.num_elites}')

    return new_elite_archive


def prepare_sample_elites(self: SailRun, new_elite_archive: GridArchive, old_elite_archive: GridArchive, pred_flag=False):
    """
    - extracts all elites from new_elite_archive
    - splits them into improved elites and new bin elites
        - (if) ucb or pred is calculated: calculates objective improvement for improved elites
        - (else) objective improvement = objective
    - orders solutions by objective improvement

    Inputs:
        Old Elite Archive
        New Elite Archive
    """

    old_elite_df = old_elite_archive.as_pandas(include_solutions=True).sort_values(by=['index'])
    new_elite_df = new_elite_archive.as_pandas(include_solutions=True).sort_values(by=['index'])

    # Index referrs to the bin, that the elite belongs to
    # Therefore the index can be used to seperate improved elites from new bin elites
    is_improved_new_elite = np.isin(new_elite_df['index'], old_elite_df['index'])
    
    improved_elites = new_elite_df[is_improved_new_elite]
    new_bin_elites   = new_elite_df[~is_improved_new_elite]

    # Select old elites that have been improved
    is_improved_old_elite = np.isin(old_elite_df['index'], improved_elites['index'])
    improved_old_elites = old_elite_df[is_improved_old_elite]

    improved_elites = improved_elites.sort_values(by=['index'])
    improved_old_elites = improved_old_elites.sort_values(by=['index'])

    new_bin_elites = new_bin_elites.assign(objective_improvement = np.array(new_bin_elites['objective'])).sort_values(by=['objective_improvement'], ascending=False)
    
    if self.acq_ucb_flag:
        improved_elites = improved_elites.assign(objective_improvement = np.array(improved_elites['objective'] - np.array(improved_old_elites['objective']))).sort_values(by=['objective_improvement'], ascending=False)

    if self.acq_mes_flag: # Dont calculate objective improvement for MES
        if pred_flag:
            improved_elites = improved_elites.assign(objective_improvement = np.array(improved_elites['objective'] - np.array(improved_old_elites['objective']))).sort_values(by=['objective_improvement'], ascending=False)
        else:
            improved_elites = improved_elites.assign(objective_improvement = np.array(improved_elites['objective'])).sort_values(by=['objective_improvement'], ascending=False)


    return improved_elites, new_bin_elites


def select_samples(self: SailRun, improved_elites, new_bin_elites, acq_flag=False, pred_flag=False, curiosity=5):
    """Selects samples based on exploration behavior defined in the class constructor"""

    if acq_flag:
        target = "Acquisition"
        n_samples = BATCH_SIZE
    if pred_flag:
        target = "Prediction"
        n_samples = PRED_N_OBJ_EVALS//PREDICTION_VERIFICATIONS

    if self.greedy_flag: # Evaluate only maximum improvement, regardeless of new/old bin
        candidate_elite_df = pandas.concat([new_bin_elites, improved_elites]).sort_values(by=['objective_improvement'], ascending=False).head(n_samples)

    if self.hybrid_flag: # Evenly balance sampling of best new_bin_elites & best improved_elites
        n_new_bin_elites = new_bin_elites.shape[0]
        n_improved_elites = improved_elites.shape[0]

        n_new_bin_samples = round((curiosity/10)*n_samples)
        n_improved_samples = n_samples - n_new_bin_samples
        if n_new_bin_elites >= n_new_bin_samples and n_improved_elites >= n_improved_samples:
            new_bin_elites = new_bin_elites.head(n_new_bin_samples)
            candidate_elite_df = pandas.concat([new_bin_elites.head(n_new_bin_samples), improved_elites.head(n_improved_samples)])
        else:
            if n_new_bin_elites < n_new_bin_samples:
                new_bin_elites = new_bin_elites.head(n_new_bin_elites)
                candidate_elite_df = pandas.concat([new_bin_elites, improved_elites.head(n_samples - n_new_bin_elites)])
            else:
                candidate_elite_df = pandas.concat([new_bin_elites.head(n_samples - n_improved_elites), improved_elites])

    return candidate_elite_df


def scale_samples(samples, boundaries=SOL_VALUE_RANGE):
    """Scales Samples to boundaries"""

    lower_bounds = np.array([boundaries[i][0] for i in range(len(boundaries))])
    upper_bounds = np.array([boundaries[i][1] for i in range(len(boundaries))])

    samples = samples * (upper_bounds - lower_bounds) + lower_bounds    
    return samples


def initialize_archive(self):

    print(f"Initialize Archive [...]")

    mes_flag = self.acq_mes_flag
    ucb_flag = self.acq_ucb_flag

    # for vanilla sail, random sail & random init just draw sobol samples
    if self.vanilla_flag or self.random_flag or self.random_init:

        self.acq_function = acq_ucb
        self.acq_mes_flag = False
        self.acq_ucb_flag = True
        self.acq_archive.set_threshold(threshold_min = ACQ_UCB_MIN_THRESHHOLD)

        # visualize empty acquisition archive
        for i in range(0, INIT_N_EVALS, BATCH_SIZE):
            self.visualize_archive(self.acq_archive, acq_flag=True)

        # initialize obj archive with sobol samples
        solution_batch = create_sobol_samples(order=INIT_N_EVALS, dim=len(SOL_VALUE_RANGE), seed=self.current_seed+5)
        solution_batch = solution_batch.T
        solution_batch = scale_samples(solution_batch)
        measures_batch = solution_batch[:, 1:3]
        eval_xfoil_loop(self, solution_batch=solution_batch, measures_batch=measures_batch, acq_flag=False)

    # use MES to fill obj archive
    if self.custom_flag and self.mes_init:

        self.acq_archive.set_threshold(threshold_min = ACQ_MES_MIN_THRESHHOLD)
        self.acq_function = acq_mes
        self.acq_mes_flag = True
        self.acq_ucb_flag = False

        # visualize empty acquisition archive
        for i in range(0, INIT_N_EVALS, BATCH_SIZE):
            self.visualize_archive(self.acq_archive, acq_flag=True)
            
        self.update_cellgrids()

        # initialize obj archive with sobol samples
        solution_batch = create_sobol_samples(order=INIT_N_EVALS, dim=len(SOL_VALUE_RANGE), seed=self.current_seed)
        solution_batch = solution_batch.T
        solution_batch = scale_samples(solution_batch)
        measures_batch = solution_batch[:, 1:3]
        eval_xfoil_loop(self, solution_batch=solution_batch, measures_batch=measures_batch, acq_flag=False)

        # initialize acq archive with sobol samples
        solution_batch = create_sobol_samples(order=500, dim=len(SOL_VALUE_RANGE), seed=self.current_seed)
        solution_batch = solution_batch.T
        solution_batch = scale_samples(solution_batch)
        measures_batch = solution_batch[:, 1:3]
        for i in range(0, 200, BATCH_SIZE):
            self.update_archive(candidate_sol=solution_batch[i:i+BATCH_SIZE], candidate_bhv=measures_batch[i:i+BATCH_SIZE], acq_flag=True)
            print(f"Initialize Acq Archive: {i+10}")

        remaining_evals = INIT_N_ACQ_EVALS
        while remaining_evals > 0:

            # calculate MES Acquisition Elites
            map_elites(self=self, acq_flag=True)
            self.visualize_archive(archive=self.acq_archive, acq_flag=True)
            acq_elites = self.acq_archive.as_pandas(include_solutions=True).sort_values(by="objective", ascending=False)

            if np.any(np.isin(acq_elites, self.sol_array).all(1)):
                raise ValueError("Duplicate Solution Error: New Solutions already exist in GP Data")

            best_elites = acq_elites.head(BATCH_SIZE)

            # combine best elites with random samples
            best_elite_objectives = np.vstack(best_elites.objective_batch())
            best_elite_solutions = np.vstack(best_elites.solution_batch())
            best_elite_measures = np.vstack(best_elites.measures_batch())

            # if best elite solutions contains duplicate rows within itself, raise an error
            if np.unique(best_elite_solutions, axis=1).shape[0] != best_elite_solutions.shape[0]:
                raise ValueError("Duplicate Solution Error: New Solutions appear twice in best_elite_solutions")

            eval_xfoil_loop(self, solution_batch=best_elite_solutions, measures_batch=best_elite_measures, acq_flag=True, candidate_targetvalues=np.vstack(best_elite_objectives))
            print(f"Best Objective: {self.obj_archive.best_elite.objective}")

            remaining_evals -= BATCH_SIZE            

    # use UCB to fill obj archive
    if self.custom_flag and self.ucb_init:

        # set to hybrid flag
        greedy_flag = self.greedy_flag
        hybrid_flag = self.hybrid_flag
        self.greedy_flag = False
        self.hybrid_flag = True

        self.acq_function = acq_ucb
        self.acq_mes_flag = False
        self.acq_ucb_flag = True

        # visualize empty acquisition archive
        for i in range(0, INIT_N_EVALS, BATCH_SIZE):
            self.visualize_archive(self.acq_archive, acq_flag=True)

        # initialize obj archive with sobol samples
        solution_batch = create_sobol_samples(order=INIT_N_EVALS, dim=len(SOL_VALUE_RANGE), seed=self.current_seed)
        solution_batch = solution_batch.T
        solution_batch = scale_samples(solution_batch)
        measures_batch = solution_batch[:, 1:3]
        eval_xfoil_loop(self, solution_batch=solution_batch, measures_batch=measures_batch, acq_flag=False)

        # initialize acq archive with sobol samples
        solution_batch = create_sobol_samples(order=50000, dim=len(SOL_VALUE_RANGE), seed=self.current_seed)
        solution_batch = solution_batch.T
        solution_batch = scale_samples(solution_batch)
        measures_batch = solution_batch[:, 1:3]
        self.update_archive(candidate_sol=solution_batch, candidate_bhv=measures_batch, acq_flag=True)          # initialize acq_archive with remaining sobol samples

        # set high initial threshold to ensure that only good solutions are added to the archive
        min_threshold = 3.5
        self.acq_archive.set_threshold(threshold_min = min_threshold)

        remaining_evals = INIT_N_ACQ_EVALS
        while remaining_evals > 0:

            # calculate UCB Acquisition Elites
            new_target_elites, _, _ = map_elites(self, acq_flag=True, new_elite_threshold=min_threshold)
            if new_target_elites.stats.num_elites < BATCH_SIZE: new_target_elites = ensure_n_new_elites(self=self, new_elite_archive=new_target_elites, acq_flag=True)   # Sample until enough new acquisition elites are found
            improved_elites, new_bin_elites = prepare_sample_elites(self=self, new_elite_archive=new_target_elites, old_elite_archive=self.obj_archive)                  # Split new_target_elites into improved elites & new bin elites, then (if self.acq_ucb_flag or pred_flag) calculate objective improvement (else) objective_improvement = objective
            candidate_solutions_df = select_samples(self, improved_elites=improved_elites, new_bin_elites=new_bin_elites, acq_flag=True, curiosity=4)                    # Select samples based on exploration behavior defined in the class constructor

            self.visualize_archive(archive=self.acq_archive, acq_flag=True)

            solution_batch = candidate_solutions_df.solution_batch()
            objective_batch = candidate_solutions_df.objective_batch()
            measures_batch = candidate_solutions_df.measures_batch()

            if np.any(np.isin(solution_batch, self.sol_array).all(1)):
                raise ValueError("Duplicate Solution Error: New Solutions already exist in GP Data")

            eval_xfoil_loop(self, solution_batch=solution_batch, measures_batch=measures_batch, acq_flag=True, candidate_targetvalues=objective_batch)
            print(f"Best Objective: {self.obj_archive.best_elite.objective}")

            remaining_evals -= BATCH_SIZE
        
        # reset to original flags
        self.greedy_flag = greedy_flag
        self.hybrid_flag = hybrid_flag

    # select acquisition function based on class constructor
    if ucb_flag:
        print("ucb entering")
        self.acq_function = acq_ucb
        self.acq_mes_flag = False
        self.acq_ucb_flag = True
        self.acq_archive.set_threshold(threshold_min = ACQ_UCB_MIN_THRESHHOLD)
    if mes_flag:
        print("mes entering")
        self.acq_function = acq_mes
        self.acq_mes_flag = True
        self.acq_ucb_flag = False
        self.acq_archive.set_threshold(threshold_min = ACQ_MES_MIN_THRESHHOLD)
        self.update_cellgrids()

    # initialize acq archive with sobol samples
    solution_batch = create_sobol_samples(order=INIT_N_SOBOL_ACQ, dim=len(SOL_VALUE_RANGE), seed=self.current_seed)
    solution_batch = solution_batch.T
    solution_batch = scale_samples(solution_batch)
    measures_batch = solution_batch[:, 1:3]

    self.acq_archive.clear()
    print(f"Acq MES Flag {self.acq_mes_flag} - Acq UCB Flag {self.acq_ucb_flag} - Acq Function {self.acq_function}")
    for i in range(0, INIT_N_SOBOL_ACQ, BATCH_SIZE):
        self.update_archive(candidate_sol=solution_batch[i:i+BATCH_SIZE], candidate_bhv=measures_batch[i:i+BATCH_SIZE], acq_flag=True)
        print(f"Initialize Acq Archive: {i+10}   Size: {self.acq_archive.stats.num_elites}")
    
    sobol_acq_elites = self.acq_archive.as_pandas(include_solutions=True)
    self.update_archive(candidate_sol=sobol_acq_elites.solution_batch(), candidate_bhv=sobol_acq_elites.measures_batch(), acq_flag=True)

    print("\n[...] Terminate init_archive()\n")
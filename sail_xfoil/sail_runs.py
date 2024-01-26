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
from acq_functions.acq_mes import optimize_mes
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
        candidate_solutions_df = self.acq_archive.as_pandas(include_solutions=True).sample(n=BATCH_SIZE, random_state=self.initial_seed, replace=False)   # ToDo: implement sobol sample as done in seed paper
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

OBJ_BHV_NUMBER_BINS = config.OBJ_BHV_NUMBER_BINS
BHV_VALUE_RANGE = config.BHV_VALUE_RANGE


def run_custom_sail(self: SailRun, acq_loop=False, pred_loop=False):
    """
    Args:
        acq_loop:   If True, will enter acquisition loop
        pred_loop:  If True, will enter prediction verification loop
    """

    if not pred_loop:
        initialize_archive(self)

    CURIOSITY = 3 # For Hybrid Approach: 'CURIOSITY//BATCH_SIZE' new bin elites are to be sampled

    anytime_metric_kwargs = initialize_anytime_metrics(self=self, acq_flag=acq_loop, pred_flag=pred_loop)

    current_eval_budget = anytime_metric_kwargs['current_eval_budget']
    consumed_obj_evals = anytime_metric_kwargs['consumed_obj_evals']
    iteration = anytime_metric_kwargs['iteration']
    total_eval_budget = anytime_metric_kwargs['total_eval_budget']
    total_iterations = total_eval_budget//BATCH_SIZE

    if acq_loop:
        i_obj_evals = BATCH_SIZE
        target_archive = self.acq_archive
    if pred_loop:
        i_obj_evals = PRED_N_OBJ_EVALS//PREDICTION_VERIFICATIONS
        target_archive = self.pred_archive

    while(current_eval_budget >= i_obj_evals):

        if iteration == total_iterations - 6:
            # Increase archive resolution
            acq_elite_df = self.acq_archive.as_pandas(include_solutions=True)
            self.acq_archive = GridArchive(
                solution_dim=SOL_DIMENSION,
                dims=[15,15],
                ranges=BHV_VALUE_RANGE,
                dtype=np.float64)
            self.update_cellgrids()
            self.update_mutant_cellgrids()
            self.update_archive(candidate_sol=acq_elite_df.solution_batch(), candidate_bhv=acq_elite_df.measures_batch(), acq_flag=True)

            # Second Archive Update
            acq_elite_df = self.acq_archive.as_pandas(include_solutions=True)
            self.update_archive(candidate_sol=acq_elite_df.solution_batch(), candidate_bhv=acq_elite_df.measures_batch(), acq_flag=True)

        if iteration == total_iterations - 3:
            # Increase archive resolution
            acq_elite_df = self.acq_archive.as_pandas(include_solutions=True)
            self.acq_archive = GridArchive(
                solution_dim=SOL_DIMENSION,
                dims=[20,20],
                ranges=BHV_VALUE_RANGE,
                dtype=np.float64)
            self.update_cellgrids()
            self.update_mutant_cellgrids()
            self.update_archive(candidate_sol=acq_elite_df.solution_batch(), candidate_bhv=acq_elite_df.measures_batch(), acq_flag=True)

            # Second Archive Update
            acq_elite_df = self.acq_archive.as_pandas(include_solutions=True)
            self.update_archive(candidate_sol=acq_elite_df.solution_batch(), candidate_bhv=acq_elite_df.measures_batch(), acq_flag=True)

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

        print("Curiosity: ", CURIOSITY)
        obj_t0, obj_t1, n_new_obj_elites = eval_xfoil_loop(self, solution_batch=solution_batch, measures_batch=measures_batch, candidate_targetvalues=objective_batch, acq_flag=acq_loop, pred_flag=pred_loop)       # Evaluate Acquisition Elites & Update Acq Archive under resulting GP Model

        # Calculate, Print & Store Anytime Metrics
        anytime_metric_kwargs = calculate_anytime_metrics(self=self, obj_t0=obj_t0, obj_t1=obj_t1, target_t0=target_t0, target_t1=target_t1, n_new_obj_elites=n_new_obj_elites, new_target_bin_elites=new_bin_elites, improved_target_elites=improved_elites, anytime_metric_kwargs=anytime_metric_kwargs, acq_flag=acq_loop, pred_flag=pred_loop)
        store_anytime_metrics(self=self, acq_flag=acq_loop, pred_flag=pred_loop, anytime_metric_kwargs=anytime_metric_kwargs)

        iteration = anytime_metric_kwargs['iteration']
        current_eval_budget = anytime_metric_kwargs['current_eval_budget']

        print(iteration)
        print(CURIOSITY)
        if iteration % 10 == 0:
            CURIOSITY += 1
            CURIOSITY = min(CURIOSITY, 7)

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
    obj_archive_df = self.obj_archive.as_pandas(include_solutions=True)
    target_archive_df = new_elite_archive.as_pandas(include_solutions=True)

    # Remove invalid designs
    valid_indices, _ = generate_parsec_coordinates(target_archive_df.solution_batch(), io_flag=False)
    target_archive_df = target_archive_df.iloc[valid_indices]

    # Remove all candidate solutions from target_archive_df, that have already been evaluated
    target_archive_df = target_archive_df[~np.isin(target_archive_df.solution_batch(), self.sol_array).all(1)]

    # Map Acquisition Elites to Objective Archive Indices 
    # (allows different archive resolutions for acq and obj archive)
    new_elite_indices = self.obj_archive.index_of(target_archive_df.measures_batch())

    # Index refers to the bin, that the elite belongs to
    # Therefore the index can be used to seperate improved elites from new bin elites
    is_improved_new_elite = np.isin(new_elite_indices, obj_archive_df['index'])

    improved_elites = target_archive_df[is_improved_new_elite]
    new_bin_elites   = target_archive_df[~is_improved_new_elite]

    # Map improved elites to objective archive indices
    improved_elites = improved_elites.assign(index = self.obj_archive.index_of(improved_elites.measures_batch()))
    new_bin_elites = new_bin_elites.assign(index = self.acq_archive.index_of(new_bin_elites.measures_batch()))

    # If duplicate indices exist, keep the one with the higher objective
    improved_elites = improved_elites.sort_values(by=['index'], ascending=False)
    improved_elites = improved_elites.drop_duplicates(subset=['index'], keep='first')

    # Select old elites that have been improved
    is_improved_old_elite = np.isin(obj_archive_df['index'], new_elite_indices)
    improved_old_elites = obj_archive_df[is_improved_old_elite]

    improved_elites = improved_elites.sort_values(by=['index'])
    improved_old_elites = improved_old_elites.sort_values(by=['index'])

    # For custom approach, calculate objective improvement, if UCB or Prediction is calculated
    if self.custom_flag and (self.acq_ucb_flag or pred_flag):
        new_bin_elites = new_bin_elites.assign(objective_improvement = np.array(new_bin_elites['objective']))
        improved_elites = improved_elites.assign(objective_improvement = np.array(improved_elites['objective'] - np.array(improved_old_elites['objective'])))

    return improved_elites, new_bin_elites


def select_samples(self: SailRun, improved_elites, new_bin_elites, acq_flag=False, pred_flag=False, curiosity=7):
    """
    - Selects samples based on exploration behavior defined in the class constructor
    - In case of MES, the best 70% of elites are selected, then randomly shuffled. This is done due to different MES value ranges
    """
    
    if acq_flag:
        n_samples = BATCH_SIZE
    if pred_flag:
        n_samples = PRED_N_OBJ_EVALS//PREDICTION_VERIFICATIONS

    if self.vanilla_flag or (self.acq_mes_flag and acq_flag):
        # shuffle elites (elites are sorted by index, which is not random)
        new_bin_elites = new_bin_elites.sample(frac=1, random_state=self.initial_seed)
        improved_elites = improved_elites.sample(frac=1, random_state=self.initial_seed)

    # For custom approach, sort by objective improvement, if UCB or Prediction is calculated
    if self.custom_flag and (self.acq_ucb_flag or pred_flag):
        new_bin_elites = new_bin_elites.sample(frac=1, random_state=self.initial_seed)
        improved_elites = improved_elites.sort_values(by=['objective_improvement'], ascending=False)

    if self.vanilla_flag:
        candidate_elite_df = pandas.concat([new_bin_elites.head(n_samples), improved_elites.head(n_samples)])
        candidate_elite_df = candidate_elite_df.sample(frac=1, random_state=self.initial_seed)

    if self.hybrid_flag: # Evenly balance sampling of best new_bin_elites & best improved_elites

        n_new_bin_elites = new_bin_elites.shape[0]
        n_improved_elites = improved_elites.shape[0]

        print(f"Available --- New Bin Elites: {n_new_bin_elites} - Improved Elites: {n_improved_elites}")

        n_new_bin_samples = round((curiosity/10)*n_samples)
        n_improved_samples = n_samples - n_new_bin_samples
        print(f"Attempting --- New Bin Elites: {n_new_bin_samples} - Improved Elites: {n_improved_samples}")

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

    # calculate initial sobol sample
    solution_batch = create_sobol_samples(order=INIT_N_EVALS, dim=len(SOL_VALUE_RANGE), seed=self.current_seed)
    solution_batch = scale_samples(solution_batch.T)
    measures_batch = solution_batch[:, 1:3]
    generate_parsec_coordinates(solution_batch)
    eval_xfoil_loop(self, solution_batch=solution_batch, measures_batch=measures_batch, acq_flag=False)

    for i in range(0, INIT_N_EVALS, BATCH_SIZE):
        self.visualize_archive(self.acq_archive, acq_flag=True)

    # for vanilla sail, random sail & random init just draw sobol samples
    if self.vanilla_flag or self.random_flag or self.random_init:

        self.acq_function = acq_ucb
        self.acq_mes_flag = False
        self.acq_ucb_flag = True

        # initialize acq archive with sobol samples
        solution_batch = create_sobol_samples(order=50000, dim=len(SOL_VALUE_RANGE), seed=self.current_seed)
        solution_batch = solution_batch.T
        solution_batch = scale_samples(solution_batch)
        measures_batch = solution_batch[:, 1:3]
        self.update_archive(candidate_sol=solution_batch, candidate_bhv=measures_batch, acq_flag=True)

    # use MES to fill obj archive
    if self.custom_flag and self.mes_init:

        self.acq_function = acq_mes
        self.acq_mes_flag = True
        self.acq_ucb_flag = False

        # fill acq archive with random solutions (will be overwritten inside mes_map_elites)
        solution_batch = create_sobol_samples(order=10, dim=len(SOL_VALUE_RANGE), seed=self.current_seed)
        solution_batch = solution_batch.T
        solution_batch = scale_samples(solution_batch)
        measures_batch = solution_batch[:, 1:3]
        self.acq_archive.add(solution_batch, np.zeros((solution_batch.shape[0]))+0.0000001, measures_batch)

        remaining_evals = INIT_N_ACQ_EVALS
        while remaining_evals > 0:

            # calculate MES Acquisition Elites
            new_elite_archive, _, _ = map_elites(self, acq_flag=True)
            improved_elites, new_bin_elites = prepare_sample_elites(self=self, new_elite_archive=new_elite_archive, old_elite_archive=self.obj_archive)
            candidate_solutions_df = select_samples(self, improved_elites=improved_elites, new_bin_elites=new_bin_elites, acq_flag=True, curiosity=7)            # Select samples based on exploration behavior defined in the class constructor

            self.visualize_archive(archive=self.acq_archive, acq_flag=True)

            solution_batch = candidate_solutions_df.solution_batch()
            objective_batch = candidate_solutions_df.objective_batch()
            measures_batch = candidate_solutions_df.measures_batch()

            eval_xfoil_loop(self, solution_batch=solution_batch, measures_batch=measures_batch, acq_flag=True, candidate_targetvalues=objective_batch)
            print(f"Best Objective: {self.obj_archive.best_elite.objective}")

            remaining_evals -= BATCH_SIZE

    # use UCB to fill obj archive
    if self.custom_flag and self.ucb_init:

        self.acq_function = acq_ucb
        self.acq_mes_flag = False
        self.acq_ucb_flag = True

        # initialize acq archive with sobol samples
        solution_batch = create_sobol_samples(order=50000, dim=len(SOL_VALUE_RANGE), seed=self.current_seed)
        solution_batch = solution_batch.T
        solution_batch = scale_samples(solution_batch)
        measures_batch = solution_batch[:, 1:3]
        self.update_archive(candidate_sol=solution_batch, candidate_bhv=measures_batch, acq_flag=True)

        remaining_evals = INIT_N_ACQ_EVALS
        while remaining_evals > 0:

            # calculate UCB Acquisition Elites
            new_elite_archive, _, _ = map_elites(self, acq_flag=True)
            improved_elites, new_bin_elites = prepare_sample_elites(self=self, new_elite_archive=new_elite_archive, old_elite_archive=self.obj_archive)                                           # Split new_target_elites into improved elites & new bin elites, then calculate acquisition improvement
            candidate_solutions_df = select_samples(self, improved_elites=improved_elites, new_bin_elites=new_bin_elites, acq_flag=True, curiosity=3)     # Select samples based on exploration behavior defined in the class constructor

            self.visualize_archive(archive=self.acq_archive, acq_flag=True)

            solution_batch = candidate_solutions_df.solution_batch()
            objective_batch = candidate_solutions_df.objective_batch()
            measures_batch = candidate_solutions_df.measures_batch()

            eval_xfoil_loop(self, solution_batch=solution_batch, measures_batch=measures_batch, acq_flag=True, candidate_targetvalues=objective_batch)
            print(f"Best Objective: {self.obj_archive.best_elite.objective}")

            remaining_evals -= BATCH_SIZE

    # select acquisition function based on class constructor
    if ucb_flag:
        print("ucb entering")
        self.acq_function = acq_ucb
        self.acq_mes_flag = False
        self.acq_ucb_flag = True
    if mes_flag:
        print("mes entering")
        self.acq_function = acq_mes
        self.acq_mes_flag = True
        self.acq_ucb_flag = False
    
    if not self.acq_mes_flag:
        acq_archive_df = self.acq_archive.as_pandas(include_solutions=True)
        acq_archive_solutions = acq_archive_df.solution_batch()
        acq_archive_measures = acq_archive_df.measures_batch()
        self.update_archive(candidate_sol=acq_archive_solutions, candidate_bhv=acq_archive_measures, acq_flag=True)
    else:
        self.update_cellgrids()
        self.update_mutant_cellgrids()

    print("\n[...] Terminate init_archive()\n")

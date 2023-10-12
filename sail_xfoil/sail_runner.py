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
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
INIT_N_EVALS = config.INIT_N_EVALS
SOL_DIMENSION = config.SOL_DIMENSION
OBJ_DIMENSION = config.OBJ_DIMENSION
BHV_DIMENSION = config.BHV_DIMENSION
ACQ_N_OBJ_EVALS = config.ACQ_N_OBJ_EVALS
ACQ_N_MAP_EVALS = config.ACQ_N_MAP_EVALS
BATCH_SIZE = config.BATCH_SIZE
PRED_N_EVALS = config.PRED_N_EVALS
BHV_NUMBER_BINS = config.BHV_NUMBER_BINS
BHV_VALUE_RANGE = config.BHV_VALUE_RANGE
SIGMA_EMITTER = config.SIGMA_EMITTER
PRED_ELITE_REEVALS = config.PRED_ELITE_REEVALS
TEST_RUNS = config.TEST_RUNS
MAX_PRED_VERIFICATION = config.MAX_PRED_VERIFICATION
SIGMA_PRED_EMITTER = config.SIGMA_PRED_EMITTER


###### Import Custom Scripts ######
from utils.anytime_archive_visualizer import anytime_archive_visualizer
from utils.pprint_nd import pprint, pprint_fstring
from utils.utils import maximize_obj_improvement, eval_xfoil_loop, scale_samples

from xfoil.generate_airfoils import generate_parsec_coordinates
from xfoil.simulate_airfoils import xfoil

from acq_functions.acq_ucb import acq_ucb
from gp.fit_gp_model import fit_gp_model
from map_elites import map_elites




class SailRun:

    def __init__(self, initial_seed, sail_vanilla_flag=False, sail_custom_flag=False, sail_random_flag=False, pred_verific_flag=False, extra_evals=None):

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

        self.current_iteration = 0
        self.extra_evals = extra_evals if extra_evals is not None else 0

        if pred_verific_flag:
            self.domain = "prediction_verification"
            self.extra_evals = 0
        if sail_custom_flag:
            self.domain = "custom"
        if sail_random_flag:
            self.domain = "random"
        if sail_vanilla_flag:
            self.domain = "vanilla"
        if pred_verific_flag:
            self.domain = self.domain + "_prediction_verification"

        self.initial_seed = initial_seed
        self.current_seed = initial_seed

        self.sol_array = np.empty((0, SOL_DIMENSION))
        self.obj_array = np.empty((0, OBJ_DIMENSION))

        print("\nInitialize SAIL Run")
        print(f"Domain: {self.domain}")
        print(f"Initial Seed: {self.initial_seed}")

        self.obj_archive, self.acq_archive, self.pred_archive = define_archives(initial_seed=initial_seed)
        obj_archive, init_solutions, init_obj_evals = initialize_archive(self, archive=self.obj_archive, seed=initial_seed)
        self.update_gp_model(init_solutions, init_obj_evals)

    def update_gp_model(self, new_solutions, new_objectives):
        self.sol_array = np.vstack((self.sol_array, new_solutions))
        self.obj_array = np.append(self.obj_array, new_objectives)
        self.obj_array = np.ravel(self.obj_array)
        self.gp_model = fit_gp_model(self.sol_array, self.obj_array)

    def update_iteration(self):
        self.current_iteration += 1

    def update_seed(self):
        self.current_seed += TEST_RUNS
        return self.current_seed

    def visualize_archive(self, archive):
        anytime_archive_visualizer(self, archive)
        self.update_iteration()

    def update_archive(self, candidate_sol=None, candidate_obj=None, candidate_bhv=None, obj_flag=False, acq_flag=False, pred_flag=False, archive=None):

        if archive is not None:
            self.archive = archive
            self.visualize_archive(self.archive)
            return

        if obj_flag:
            print("Elites in obj_archive (before): " + str(self.obj_archive.stats.num_elites))
            status_vector, _ = self.obj_archive.add(candidate_sol, candidate_obj, candidate_bhv)
            print("Elites in obj_archive  (after): " + str(self.obj_archive.stats.num_elites))
            self.visualize_archive(self.obj_archive)
            return status_vector, self.obj_archive
        if acq_flag:
            print("Elites in acq_archive (before): " + str(self.acq_archive.stats.num_elites))
            status_vector, _ = self.acq_archive.add(candidate_sol, candidate_obj, candidate_bhv)
            print("Elites in acq_archive  (after): " + str(self.acq_archive.stats.num_elites))
            self.visualize_archive(self.acq_archive)
            return status_vector, self.acq_archive
        if pred_flag:
            print("Elites in pred_archive (before): " + str(self.pred_archive.stats.num_elites))
            status_vector, _ = self.pred_archive.add(candidate_sol, candidate_obj, candidate_bhv)
            print("Elites in pred_archive  (after): " + str(self.pred_archive.stats.num_elites))
            self.visualize_archive(self.pred_archive)
            return status_vector, self.pred_archive
        






def initialize_archive(self: SailRun, archive: GridArchive, seed: int):

    print("\nInitialize init_archive() [...]")
    n_evals = INIT_N_EVALS
    while n_evals >= BATCH_SIZE:
        n_evals -= BATCH_SIZE
        solutions = np.empty((0, SOL_DIMENSION))
        bhv_evals = np.empty((0, BHV_DIMENSION))
        samples = create_sobol_samples(order=BATCH_SIZE, dim=len(SOL_VALUE_RANGE), seed=seed)
        samples = samples.T

        self.update_seed()

        scale_samples(samples)                  # sobol samples produce values between [0;1]
        valid_indices, surface_area_batch = generate_parsec_coordinates(samples)
        convergence_errors, success_indices, obj_batch = xfoil(iterations=BATCH_SIZE)

        sol_batch = samples[success_indices]
        bhv_batch = np.array([samples[success_indices,1:3]])
        bhv_batch = bhv_batch.reshape(10, 2)
        self.update_archive(candidate_sol=sol_batch,candidate_obj=obj_batch, candidate_bhv=bhv_batch, obj_flag=True)

    print("[...] Terminate init_archive()\n")
    return archive, sol_batch, obj_batch



def run_custom_sail(self: SailRun):
    """
    Extra evaluations are given if eval_pred_flag is True (see sail.py)
        if not eval_pred_flag, extra_evals = 0
    """

    iteration = 1
    mean_acq = 0
    mean_obj = 0
    total_obj_improvements = 0
    total_convergence_errors = 0
    mean_acq_improvement = 0
    anytime_metrics = pandas.DataFrame(columns=['Iteration', 'Mean Obj', 'Mean Acq', 'Mean Obj Improvement', 'Mean Acq Improvement', 'Percentage Improvements', 'Total Improvements', 'Percentage Convergence Errors', 'Convergence Errors', 'Total Convergence Errors', 'New Acq Elites', 'New Obj Elites'])
    eval_budget = ACQ_N_OBJ_EVALS + self.extra_evals

    # dummy value for update_emitter function, to not extract using list comprehension in every while loop
    dummy_elites = [elite.solution for elite in self.obj_archive][:BATCH_SIZE] 

    while(eval_budget >= BATCH_SIZE):

        old_elites = np.array([(  elite.solution,     elite.index,     elite.objective,       elite.measures) for elite in self.acq_archive],
                        dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])

        # Seperate function compared to Class Method update_archive()
        # Combines obj_archive and acq_archive into one archive (used for benchmarking differences)
        acq_archive = update_acq_archive(self.acq_archive, self.obj_archive, self.gp_model)
        acq_emitter = update_emitter(self, self.acq_archive, dummy_elites)

        # only used for rendering imgs
        self.update_archive(archive=acq_archive, acq_flag=True)
        
        seed_t0 = self.current_seed
        self.update_seed()
        seed_t1 = self.current_seed

        if seed_t0 == seed_t1:
            raise ValueError("Seed not updated")

        t0_acq_archive = self.acq_archive.stats.num_elites
        target_archive, new_elite_archive = map_elites(self, target_archive=self.acq_archive, emitter=acq_emitter, n_evals=ACQ_N_MAP_EVALS, fuct_obj=acq_ucb)
        t1_acq_archive = self.acq_archive.stats.num_elites

        # maximize_obj_improvement returns elites sorted in descending order (acq is obj within acq loop)
        improved_elites, new_elites, n_improvements = maximize_obj_improvement(new_elite_archive, old_elites) 

        # evaluate improved_elites & new_elites
        # update obj_archive and gp_model inside eval_max_obj_improvement()
        new_elite_sol, new_elite_obj, new_elite_bhv, acq_archive, gp_model = eval_max_obj_improvement(self, improved_elites, new_elites, old_elites, n_obj_evals=BATCH_SIZE, 
                                                                               emitter=acq_emitter, target_archive=acq_archive, n_map_evals=BATCH_SIZE, 
                                                                               fuct_obj=acq_ucb, explore_flag=True, greedy_flag=False, new_elite_archive=new_elite_archive)

        eval_budget -= BATCH_SIZE
        iteration += 1
        
        # Calculate input values for Anytime Metrics
        acquisition_values = np.vstack(improved_elites['objective'],new_elites['objective'])
        acquisition_improvements = np.vstack(improved_elites['objective_improvement'],new_elites['objective_improvement'])
        old_elite_obj = old_elites['objective']
        new_elite_obj = new_elites['objective']
        convergence_errors = BATCH_SIZE - new_elite_obj.shape[0]
        t0_obj_archive = self.obj_archive.stats.num_elites
        status_vector = self.update_archive(new_elite_sol, new_elite_obj, new_elite_bhv)
        t1_obj_archive = self.obj_archive.stats.num_elites
        # Calculate Anytime Metrics
        total_obj_improvements += new_elite_obj.shape[0]
        total_convergence_errors += BATCH_SIZE - new_elite_sol.shape[0]
        percentage_improvements       = (total_obj_improvements / (ACQ_N_OBJ_EVALS + self.extra_evals - eval_budget))*100
        percentage_convergence_errors = (total_convergence_errors / (ACQ_N_OBJ_EVALS + self.extra_evals - eval_budget))*100
        mean_acq_improvement = (mean_acq_improvement + np.mean(acquisition_improvements))/2
        mean_obj_improvement = (mean_obj_improvement + np.mean(old_elite_obj - new_elite_obj))/2
        mean_acq += np.mean(acquisition_values)
        mean_obj += np.mean(new_elite_obj)
        new_acq_elites = t1_acq_archive - t0_acq_archive
        new_obj_elites = t1_obj_archive - t0_obj_archive
        # Print Anytime Metrics
        print("Total Improvements: "       + str(total_obj_improvements))
        print("Total Convergence Errors: " + str(total_convergence_errors))
        print("Percentage Improvements: {:.1f}".format(percentage_improvements) + "%")
        print("Percentage Convergence Errors: {:.1f}".format(percentage_convergence_errors) + "%")
        print("Mean Acq Improvement: {:.1f}".format(mean_acq_improvement))
        print("Mean Obj Improvement: {:.1f}".format(mean_obj_improvement))
        print("   Acq Archive Size (before): " + str(t0_acq_archive))
        print("   Acq Archive Size  (after): " + str(t1_acq_archive))
        print("   New Acq Elites: " + str(new_acq_elites))
        print("   Obj Archive Size (before): " + str(t0_obj_archive))
        print("   Obj Archive Size  (after): " + str(t1_obj_archive))
        print("   New Obj Elites: " + str(new_obj_elites))
        print("Airfoil Convergence Errors: " + str(convergence_errors))
        print("Remaining ACQ Precise Evals: " + str(eval_budget) + "\n\n")
        # Store Anytime Metrics in Pandas Dataframe
        anytime_data = [iteration, mean_obj, mean_acq, mean_obj_improvement, mean_acq_improvement, percentage_improvements, total_obj_improvements, percentage_convergence_errors, convergence_errors, total_convergence_errors, new_acq_elites, new_obj_elites]
        anytime_metrics.loc[len(anytime_metrics)] = anytime_data

        if iteration % 20 == 0:
            # Save Anytime Metrics to CSV. If no csv is saved create new csv, else append to existing csv
            try:
                anytime_metrics.to_csv('anytime_metrics.csv', mode='a', header=False, index=False)
            except:
                anytime_metrics.to_csv('anytime_metrics.csv', index=False)
            # Reset to empty dataframe
            anytime_metrics = pandas.DataFrame(columns=['Iteration', 'Mean Obj', 'Mean Acq', 'Mean Obj Improvement', 'Mean Acq Improvement', 'Percentage Improvements', 'Total Improvements', 'Percentage Convergence Errors', 'Convergence Errors', 'Total Convergence Errors', 'New Acq Elites', 'New Obj Elites'])

        # check status vector sum
        sum_status_vector = np.sum(status_vector > 0)
        if sum_status_vector != t1_obj_archive-t0_obj_archive:
            raise ValueError("Status Vector not equal to difference in archive size")

    return


def run_vanilla_sail(acq_archive, obj_archive, gp_model, sol_array, obj_array, extra_evals, initial_seed, benchmark_domain):
    """
    Extra evaluations are given if eval_pred_flag is True (see sail.py)
        if not eval_pred_flag, extra_evals = 0
    """

    print(f"\n\nExtra evaluations (input): {extra_evals}\n\n")


    acq_emitter = update_emitter(obj_archive, acq_archive, gp_model, seed=0)

    total_improvements = 0
    total_convergence_errors = 0
    mean_acq_improvement = 0
    eval_budget = ACQ_N_OBJ_EVALS + extra_evals
    while(eval_budget >= BATCH_SIZE):
        eval_budget -= BATCH_SIZE

        # store best elites from obj_archive in acq_archive & update acquisition values
        #acq_archive = store_n_best_elites(acq_archive, obj_archive.stats.num_elites, update_acq=True, gp_model=gp_model, obj_archive=obj_archive)
        acq_archive, _ = map_elites(acq_archive, acq_emitter, gp_model, ACQ_N_MAP_EVALS, acq_ucb)                           # evolve acquisition archive

        acq_elite_batch = acq_archive.sample_elites(BATCH_SIZE)        
        acq_elite_solutions = acq_elite_batch[0]
        acq_elite_acquisitions = acq_elite_batch[1]
        acq_elite_measures = acq_elite_batch[2]

        # acq_archive only contains valid solutions (= non-intersecting polynomials), therefore no need to check for validity using valid_indices
        _, surface_area_batch = generate_parsec_coordinates(acq_elite_solutions)
        convergence_errors, success_indices, obj_batch = xfoil(BATCH_SIZE)

        # store evaluations for GP model
        sol_array = np.vstack((sol_array, acq_elite_solutions[success_indices])) # dtype=float64
        obj_array = np.vstack((obj_array, obj_batch.reshape(-1,1))) # dtype=float64

        acq_batch = acq_elite_acquisitions[success_indices]
        status_vector, value_vector = obj_archive.add(acq_elite_solutions[success_indices], obj_batch, acq_elite_measures[success_indices])
        total_improvements += np.sum(status_vector > 0)
        total_convergence_errors += np.sum(convergence_errors)
        mean_obj_improvement = np.mean(obj_batch)/(BATCH_SIZE-convergence_errors)
        percentage_improvements = (total_improvements/(ACQ_N_OBJ_EVALS-eval_budget))*100
        percentage_convergence_errors = (total_convergence_errors/(ACQ_N_OBJ_EVALS-eval_budget))*100

        print("Total Improvements: " + str(total_improvements))
        print("Total Convergence Errors: " + str(total_convergence_errors))
        print("Percentage Improvements: {:.1f}".format(percentage_improvements) + "%")
        print("Percentage Convergence Errors: {:.1f}".format(percentage_convergence_errors) + "%")
        print("Mean Obj Improvement: {:.1f}".format(mean_obj_improvement))
        print("Status Vector: " + str(status_vector))
        pprint_fstring(acq_batch, obj_batch)
        print("Acq Archive Size: " + str(acq_archive.stats.num_elites))
        print("Obj Archive Size: " + str(obj_archive.stats.num_elites))
        print("Airfoil Convergence Errors: " + str(convergence_errors))
        print("Remaining ACQ Precise Evals: " + str(eval_budget) + "\n\n")

        anytime_archive_visualizer(archive=obj_archive, benchmark_domain=benchmark_domain, initial_seed=initial_seed, iteration=(ACQ_N_OBJ_EVALS+extra_evals-eval_budget)//BATCH_SIZE)

        gp_model = fit_gp_model(sol_array, obj_array)

    return obj_archive, gp_model


def run_random_sail(acq_archive, obj_archive, gp_model, sol_array, obj_array, extra_evals, initial_seed, benchmark_domain):
    """
    Extra evaluations are given if eval_pred_flag is True (see sail.py)
        if not eval_pred_flag, extra_evals = 0
    """


    eval_budget = ACQ_N_OBJ_EVALS + extra_evals
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

        obj_archive.add(converged_samples, obj_batch, converged_behavior)

        sol_array = np.vstack((sol_array, converged_samples)) # dtype=float64
        obj_array = np.vstack((obj_array, obj_batch.reshape(-1,1))) # dtype=float64

        eval_budget -= BATCH_SIZE
        anytime_archive_visualizer(archive=obj_archive, benchmark_domain=benchmark_domain, initial_seed=initial_seed, iteration=(ACQ_N_OBJ_EVALS+extra_evals-eval_budget)//BATCH_SIZE)

        pprint(obj_batch)
        print("\n\nAirfoil Convergence Errors: " + str(convergence_errors))
        print("Remaining Random Search Obj Evals: " + str(eval_budget) + "\n\n")

    return obj_archive, gp_model


def eval_max_obj_improvement(self: SailRun, improved_elites, new_elites, old_elites, n_obj_evals, emitter, target_archive,  n_map_evals, fuct_obj, new_elite_archive, greedy_flag=False, explore_flag=False, acq_flag=False):
    
    n_obj_improvements = improved_elites.shape[0] + new_elites.shape[0]
    # Evaluate until minimum BATCH_SIZE improvements are found
    if n_obj_improvements < n_obj_evals:
        iter = 0
        while n_obj_improvements < n_obj_evals and iter <= 4:

            print("\n\n### Not enough Acq Improvements: Re-entering acquisition loop###\n\n")
            self.acq_archive, new_elite_archive = map_elites(self, emitter=emitter, target_archive=target_archive, n_evals=n_map_evals, fuct_obj=fuct_obj, new_elite_archive=new_elite_archive)
            iter += 1

            max_improvement_elites, new_elites, n_improvements = maximize_obj_improvement(new_elite_archive, old_elites)

    if explore_flag:
        # Evaluate new_elites first
        if new_elites.shape[0] >= n_obj_evals:
            candidate_elites = new_elites[:n_obj_evals]
        else:
            candidate_elites = np.concatenate((new_elites, max_improvement_elites), axis=0)
            candidate_elites = candidate_elites[:n_obj_evals]

    if greedy_flag:
        # Evaluate max_improvement_elites first
        if max_improvement_elites.shape[0] >= n_obj_evals:
            candidate_elites = new_elites[:n_obj_evals]
        else:
            candidate_elites = np.concatenate((new_elites, max_improvement_elites), axis=0)
            candidate_elites = candidate_elites[:n_obj_evals]

    candidate_sol = candidate_elites['solution']
    candidate_obj = candidate_elites['objective_improvement'] # obj = acq in acq loop
    candidate_bhv = candidate_elites['behavior']

    n_candidate_elites = candidate_sol.shape[0]
    if n_candidate_elites > n_obj_evals:
        raise ValueError("Candidate elites should be less than or equal to BATCH_SIZE")

    # acq_archive only contains valid solutions (= non-intersecting polynomials), therefore no need to check for validity using valid_indices
    conv_sol_batch, conv_obj_batch, conv_bhv_batch, succes_indices_batch, archive, extra_evals = eval_xfoil_loop(self, candidate_sol, candidate_bhv, self.extra_evals, archive=None)
    self.update_gp_model(conv_sol_batch, conv_obj_batch)

    # if elite didnt converge, remove from acq_archive
    if acq_flag:
        
        acq_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = -1,
            seed=self.current_seed
        )

        acq_archive.clear()
        obj_elites = [(elite.solution, elite.objective, elite.measures) for elite in self.obj_archive]
        acq_archive.add(obj_elites[0], obj_elites[1], obj_elites[2])
        acq_archive.add(conv_sol_batch, conv_obj_batch, conv_bhv_batch)

    archive = self.update_archive(conv_sol_batch, conv_obj_batch, conv_bhv_batch, obj_flag=True)
    gp_model = self.update_gp_model(candidate_sol, conv_obj_batch)

    return candidate_sol, candidate_obj, candidate_bhv, archive, gp_model


def update_emitter(self: SailRun, archive, dummy_elites):
    """
    Input: Updated Archive, seed
    Output: Gaussian Emitter
    """

    emitter = [
        GaussianEmitter(
        archive=archive,
        sigma=SIGMA_EMITTER,
        bounds= np.array(SOL_VALUE_RANGE),
        batch_size=BATCH_SIZE,
        initial_solutions=dummy_elites, # these solutions are never used, as the archive is never empty - however, specification is required for initializing the GaussianEmitter class
        seed=self.current_seed
    )]

    return emitter


def update_acq_archive(acq_archive, obj_archive, gp_model):
    """
    Seperate function compared to Class Method update_archive()
    Combines obj_archive and acq_archive into one archive (used for benchmarking differences)
    Also, calling functions within the module is faster

        Input: Updated Archive, GP Model
        Output: Updated Archive
    """

    n_obj_elites = sorted(obj_archive, key=lambda x: x.objective, reverse=True)[:obj_archive.stats.num_elites]
    n_acq_elites = sorted(acq_archive, key=lambda x: x.objective, reverse=True)[:acq_archive.stats.num_elites]

    n_obj_sol = np.array([elite.solution for elite in n_obj_elites])
    n_acq_sol = np.array([elite.solution for elite in n_acq_elites])

    n_obj_acq = np.array([elite.objective for elite in n_obj_elites])
    n_acq_acq = acq_ucb(n_acq_sol, gp_model) if n_acq_sol.shape[0] > 0 else np.array([])

    n_obj_bhv = np.array([elite.measures for elite in n_obj_elites])
    n_acq_bhv = np.array([elite.measures for elite in n_acq_elites])

    n_sol = np.concatenate((n_obj_sol, n_acq_sol), axis=0) if n_acq_sol.shape[0] > 0 else n_obj_sol
    n_acq = np.concatenate((n_obj_acq, n_acq_acq), axis=0) if n_acq_acq.shape[0] > 0 else n_obj_acq
    n_bhv = np.concatenate((n_obj_bhv, n_acq_bhv), axis=0) if n_acq_bhv.shape[0] > 0 else n_obj_bhv

    acq_archive.clear()
    acq_archive.add(n_sol, n_acq, n_bhv)

    return acq_archive


def define_archives(initial_seed):
    """Reduces Overhead"""

    obj_archive = GridArchive(
        solution_dim=SOL_DIMENSION,         # Dimension of solution vector
        dims=BHV_NUMBER_BINS,               # Discretization of behavioral bins
        ranges=BHV_VALUE_RANGE,             # Possible values for behavior vector
        qd_score_offset=-600,
        threshold_min = -1,
        seed=initial_seed
        )
    
    acq_archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = -1,
        seed=initial_seed
        )
    
    pred_archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = -1,
        seed=initial_seed,
        )
    
    return obj_archive, acq_archive, pred_archive
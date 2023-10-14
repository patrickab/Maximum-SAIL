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

PRED_ELITE_REEVALS = config.PRED_ELITE_REEVALS
MAX_PRED_VERIFICATION = config.MAX_PRED_VERIFICATION
PREDICTION_VERIFICATIONS = config.PREDICTION_VERIFICATIONS

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

    def __init__(self, initial_seed, sail_vanilla_flag=False, sail_custom_flag=False, sail_random_flag=False, pred_verific_flag=False, greedy_flag=False, explore_flag=False, extra_evals=0):

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
        self.extra_evals = extra_evals

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
        if pred_verific_flag:
            self.domain = self.domain + "_prediction_verification"
        if greedy_flag and explore_flag:
            raise ValueError("Greedy and Explore Flags cannot both be True")
        if pred_verific_flag and not (greedy_flag or explore_flag):
            raise ValueError("Prediction Verification Flag requires Greedy or Explore Flag to be True")

        # stores new solutions from custom_update_archive()
        self.new_sol = np.empty((0, SOL_DIMENSION))
        self.new_obj = np.empty((0, OBJ_DIMENSION))
        self.new_bhv = np.empty((0, BHV_DIMENSION))

        self.initial_seed = initial_seed
        self.current_seed = initial_seed

        self.greedy_flag = greedy_flag
        self.explore_flag = explore_flag
        self.pred_verific_flag = pred_verific_flag

        self.sol_array = np.empty((0, SOL_DIMENSION))
        self.obj_array = np.empty((0, OBJ_DIMENSION))

        self.obj_archive, self.acq_archive, self.pred_archive = self.define_archives(initial_seed)

        print("\nInitialize SAIL Run")
        print(f"Domain: {self.domain}")
        print(f"Initial Seed: {self.initial_seed}")    
        print(f"Initialize Archive [...]")
        total_errors = 0

        samples = create_sobol_samples(order=INIT_N_EVALS, dim=len(SOL_VALUE_RANGE), seed=self.current_seed)
        samples = samples.T
        scale_samples(samples) # sobol samples are between [0;1]
        self.init_samples = samples
        eval_xfoil_loop(self, samples, pred_flag=False, acq_flag=False)   # fill obj archive inside eval_xfoil_loop()
        self.update_archive(self.new_sol, self.new_obj, self.new_bhv, acq_flag=True)   # initialize acq_archive with obj_elites

        # If the # of convergence errors doesnt match the # of converged solutions
        if self.n_errors != INIT_N_EVALS-self.sol_array.shape[0]:
            raise ValueError("Archive Initialization Error")

        print("[...] Terminate init_archive()\n")

    
    def update_gp_data(self, new_solutions, new_objectives):

        print(f"Update GP Data [...]")
        pprint(new_solutions, new_objectives)
        n_new = new_solutions.shape[0]
        n_old = self.sol_array.shape[0]
        n_expected = n_old + n_new 
        # np.vstack x and y for bulletproof functionality
        new_solutions = np.vstack(new_solutions) if new_solutions.shape[0] != 0 else new_solutions
        new_objectives = np.vstack(new_objectives) if new_solutions.shape[0] != 0 else new_objectives
        self.sol_array = np.vstack((self.sol_array, new_solutions))
        self.obj_array = np.vstack((self.obj_array, new_objectives))
        n_resulted = self.sol_array.shape[0]
        print(f"GP Data Points: {n_resulted}")

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

    def update_iteration(self):
        self.current_iteration += 1

    def update_seed(self):

        self.current_seed += TEST_RUNS
        return self.current_seed

    def visualize_archive(self, archive):
        anytime_archive_visualizer(self, archive)
        self.update_iteration()

    def update_archive(self, candidate_sol=None, candidate_obj=None, candidate_bhv=None, obj_flag=False, acq_flag=False, pred_flag=False, surpress_print=False):
        """"
        Input:
            Option 1: Call with archive & archive flag
            Option 2: Call with candidate_sol, candidate_obj, candidate_bhv & archive flag
        """            

        candidate_obj = candidate_obj.ravel()

        if np.sum([obj_flag, acq_flag, pred_flag] == True) > 1:
            raise ValueError("More than one flag is True, use update_acq_archive")
        
        if obj_flag:

            self.obj_t0 = self.obj_archive.stats.num_elites    
            status_vector, _ = self.obj_archive.add(candidate_sol, candidate_obj, candidate_bhv)
            non_0_status_indices = np.where(status_vector != 0)[0]
            self.new_sol = candidate_sol[non_0_status_indices]
            self.new_obj = candidate_obj[non_0_status_indices]
            self.new_bhv = candidate_bhv[non_0_status_indices]
            self.obj_t1 = self.obj_archive.stats.num_elites
            self.visualize_archive(self.obj_archive)
            self.convergence_errors = BATCH_SIZE - candidate_sol.shape[0]
            if not surpress_print:
                print("Elites in Obj Archive (before): " + str(self.obj_t0))
                print("Elites in Obj Archive  (after): " + str(self.obj_t1))


        if acq_flag:

            self.acq_t0 = self.acq_archive.stats.num_elites
            status_vector, _ = self.acq_archive.add(candidate_sol, candidate_obj, candidate_bhv)
            non_0_status_indices = np.where(status_vector != 0)[0]

            if self.pred_verific_flag:
                # new target sol is set in custom_update_archive()
                # defining update_archive in this manner enables us to
                # use update_archive() regardless of pred_verific_flag
                self.custom_update_archive(self.acq_archive, acq_ucb, acq_flag=True) # modularize later to use MES
                #self.visualize_archive(self.acq_archive)
                self.acq_t1 = self.acq_archive.stats.num_elites
                if not surpress_print:
                    print("Elites in Acq Archive (before): " + str(self.acq_t0))
                    print("Elites in Acq Archive  (after): " + str(self.acq_t1))  

            else :
                #self.visualize_archive(self.acq_archive)
                self.acq_t1 = self.acq_archive.stats.num_elites
                self.new_target_elite_sol = candidate_sol[non_0_status_indices]
                self.new_target_elite_obj = candidate_obj[non_0_status_indices]
                self.new_target_elite_bhv = candidate_bhv[non_0_status_indices]
                if not surpress_print:
                    print("Elites in Acq Archive (before): " + str(self.acq_t0))
                    print("Elites in Acq Archive  (after): " + str(self.acq_t1))  


        if pred_flag:

            self.pred_t0 = self.pred_archive.stats.num_elites
            status_vector, _ = self.pred_archive.add(candidate_sol, candidate_obj, candidate_bhv)
            non_0_status_indices = np.where(status_vector != 0)[0]

            if self.pred_verific_flag:
                # new target sol is set in custom_update_archive()
                # defining update_archive in this manner enables us to
                # use update_archive() regardless of pred_verific_flag
                self.custom_update_archive(self.pred_archive, predict_objective, pred_flag=True)
                #self.visualize_archive(self.pred_archive)
                self.pred_t1 = self.pred_archive.stats.num_elites
                if not surpress_print:
                    print("Elites in Pred Archive (before): " + str(self.pred_t0))
                    print("Elites in Pred Archive  (after): " + str(self.pred_t1))  

            else :
                #self.visualize_archive(self.pred_archive)
                self.pred_t1 = self.pred_archive.stats.num_elites
                self.new_target_elite_sol = candidate_sol[non_0_status_indices]
                self.new_target_elite_obj = candidate_obj[non_0_status_indices]
                self.new_target_elite_bhv = candidate_bhv[non_0_status_indices]
                if not surpress_print:
                    print("Elites in Pred Archive (before): " + str(self.pred_t0))
                    print("Elites in Pred Archive  (after): " + str(self.pred_t1))          




    def custom_update_archive(self, target_archive ,target_function, pred_flag=False, acq_flag=False):
        """

        Seperate function compared to Class Method update_archive()
        Combines obj_archive and ((archive)) into one archive 
        to ensure (minimum fitness == objective archive fitness).

        This function also preserves elites in the (prediction/acquisition) archive,
        that remain highly performant even after refitting the Gaussian Process.

        This is done by reevaluating fuct_acq() in acq_elites, 
        and then letting them compete against obj_elites.

        New elites are stored in self.new_* class variables

        Make sure to call this function right after model refitting.
    
            Input: Updated Archive, GP Model, Flag for model fitting
                   target_function = fuct_acq() or fuct_predict(), in order to update acquisition values or predictions according to new GP
            Output: Updated Archive
        """
        if pred_flag:
            self.pred_archive = target_archive
        if acq_flag:
            self.acq_archive = target_archive

        # extract elites from obj_archive
        n_obj_elites = sorted(self.obj_archive, key=lambda x: x.objective, reverse=True)[:self.obj_archive.stats.num_elites]
        n_obj_sol = np.array([elite.solution for elite in n_obj_elites])    
        n_obj_bhv = np.array([elite.measures for elite in n_obj_elites])
        n_obj_acq = np.array([elite.objective for elite in n_obj_elites])

        # extract & reevaluate elites from target_archive
        target_elites = sorted(target_archive, key=lambda x: x.objective, reverse=True)[:target_archive.stats.num_elites]
        target_elite_obj = np.array([elite.solution for elite in target_elites])
        target_elite_bhv = np.array([elite.measures for elite in target_elites])
        target_elite_acq = target_function(target_elite_obj, self.gp_model) # cant dynamically program this for some reason, therefore acq_ucb is set as default
    
        # concatenate elites from both archives
        n_sol = np.concatenate((n_obj_sol, target_elite_obj), axis=0) if target_archive.stats.num_elites != 0 else n_obj_sol
        n_acq = np.concatenate((n_obj_acq, target_elite_acq), axis=0) if target_archive.stats.num_elites != 0 else n_obj_acq
        n_bhv = np.concatenate((n_obj_bhv, target_elite_bhv), axis=0) if target_archive.stats.num_elites != 0 else n_obj_bhv
    
        # let elites compete, store resulting archive
        if acq_flag:
            self.acq_t0 = self.acq_archive.stats.num_elites
            self.acq_archive.clear()
            status_vector, _ = self.acq_archive.add(n_sol, n_acq, n_bhv)
            self.acq_t1 = self.acq_archive.stats.num_elites
        if pred_flag:
            self.pred_t0 = self.pred_archive.stats.num_elites
            self.pred_archive.clear()
            status_vector, _ = self.pred_archive.add(n_sol, n_acq, n_bhv)
            self.pred_t1 = self.pred_archive.stats.num_elites

        self.new_target_elite_sol = n_sol[status_vector]
        self.new_target_elite_obj = n_acq[status_vector]
        self.new_target_elite_bhv = n_bhv[status_vector]
        return
    

    def set_n_errors(self, n_errors):
        self.n_errors = n_errors
        return
    

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

        return obj_archive, acq_archive, pred_archive


def run_custom_sail(self: SailRun):
    """
    Extra evaluations are given if eval_pred_flag is True (see sail.py)
        if not eval_pred_flag, extra_evals = 0
    """

    iteration = 1
    mean_obj = 0
    mean_acq = 0
    i_mean_acq = 0
    i_mean_obj = 0
    total_obj_improvements = 0
    total_acq_improvements = 0
    total_convergence_errors = 0
    mean_acq_improvement = 0
    mean_obj_improvement = 0 # ToDo
    anytime_metrics = pandas.DataFrame(columns=['Iteration', 'Mean Obj', 'Mean Acq', 'Mean Obj Improvement', 'Mean Acq Improvement', 'Percentage Improvements', 'Total Improvements', 'Percentage Convergence Errors', 'Convergence Errors', 'Total Convergence Errors', 'New Acq Elites', 'New Obj Elites'])
    eval_budget = ACQ_N_OBJ_EVALS

    while(eval_budget >= BATCH_SIZE):

        old_acq_elites = np.array([(  elite.solution,     elite.index,     elite.objective,       elite.measures) for elite in self.acq_archive],
                        dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])


        # calculate & evaluate elites, that maximize acquisition values
        # update obj_archive and gp_model inside eval_max_obj_improvement()
        map_elites(self, target_archive=self.acq_archive, target_function=acq_ucb, acq_flag=True)
        improved_elites, new_bin_elites = maximize_improvement(new_elite_archive=self.acq_archive, old_elites=old_acq_elites) 
        evaluate_max_improvement(self, improved_elites=improved_elites, new_bin_elites=new_bin_elites, old_elites=old_acq_elites, target_function=acq_ucb, acq_flag=True)

        eval_budget -= BATCH_SIZE
        
        # Calculate Anytime Metrics
        acquisition_improvements = np.vstack(np.hstack((improved_elites['objective_improvement'],new_bin_elites['objective_improvement'])))

        convergence_errors = self.convergence_errors

        # Mean Obj & Acq in current iteration
        i_mean_obj += np.mean(self.new_obj)
        i_mean_acq += np.mean(self.new_target_elite_obj)
        mean_obj = np.mean(mean_obj + i_mean_obj)
        mean_acq = np.mean(mean_acq + i_mean_acq)
        mean_acq_improvement = np.mean(acquisition_improvements)

        # Elites in archive before & after
        obj_t0 = self.obj_t0
        obj_t1 = self.obj_t1
        acq_t0 = self.acq_t0
        acq_t1 = self.acq_t1
        n_obj_improvements = obj_t1 - obj_t0
        n_acq_improvements = acq_t1 - acq_t0
        total_obj_improvements += n_obj_improvements
        total_acq_improvements += n_acq_improvements
        percentage_improvements       = (total_obj_improvements/ACQ_N_OBJ_EVALS)*100
        percentage_convergence_errors = (total_convergence_errors/ACQ_N_OBJ_EVALS)*100

        # Acq/Obj Metrics
        qd_obj = self.obj_archive.stats.qd_score
        qd_acq = self.acq_archive.stats.qd_score
        print(self.obj_archive.dims)
        obj_qd_per_bin = qd_obj/self.obj_archive.dims
        acq_qd_per_bin = qd_acq/self.acq_archive.dims
        obj_mean_qd_score = qd_obj/self.obj_archive.stats.num_elites
        acq_mean_qd_score = qd_acq/self.acq_archive.stats.num_elites
        # ToDo: mean improvements (new_converged_elites - old_corresponding_elites) / n_converged_elites 
        # ToDo: print new_converged_elites next to old_corresponding_elites
        # old_corresponding_elites -> right join by index
        # print("Mean Obj Improvement: {:.1f}".format(mean_obj_improvement))
        old_elite_obj = old_acq_elites['objective']

        # Print Anytime Metrics
        print("Percentage Improvements: {:.1f}".format(percentage_improvements) + "%")
        print("Percentage Convergence Errors: {:.1f}".format(percentage_convergence_errors) + "%")
        print("Total Improvements: "       + str(total_obj_improvements))
        print("Total Convergence Errors: " + str(total_convergence_errors))
        print("Mean Acq Improvement: {:.1f}".format(mean_acq_improvement))
        print("   Acq Archive Size (before): " + str(acq_t0))
        print("   Acq Archive Size  (after): " + str(acq_t1))
        print("   Mean Acq QD Score: " + str(acq_qd_per_bin))        
        print("   New Acq Elites: " + str(acq_t1 - acq_t0))
        print("   Acq QD Score:   " + str(acq_mean_qd_score))
        print("   Obj QD Score:   " + str(obj_mean_qd_score))
        print("   New Obj Elites: " + str(obj_t1 - obj_t0))
        print("   Mean Obj QD Score: " + str(obj_qd_per_bin))
        print("   Obj Archive Size (before): " + str(obj_t0))
        print("   Obj Archive Size  (after): " + str(obj_t1))
        print("Airfoil Convergence Errors: " + str(convergence_errors))
        print("Remaining ACQ Precise Evals: " + str(eval_budget) + "\n\n")
        # Store Anytime Metrics in Pandas Dataframe
        anytime_data = [iteration, i_mean_obj, obj_qd_per_bin, obj_mean_qd_score,
                                   i_mean_acq, acq_qd_per_bin, acq_mean_qd_score,
                                   percentage_improvements, total_obj_improvements,
                                   percentage_convergence_errors, convergence_errors,
                                   total_convergence_errors]
        anytime_metrics.loc[len(anytime_metrics)] = anytime_data

        if iteration % 20 == 0:
            # Save Anytime Metrics to CSV. If no csv is saved create new csv, else append to existing csv
            try:
                anytime_metrics.to_csv('acq_loop_anytime_metrics.csv', mode='a', header=False, index=False)
            except:
                anytime_metrics.to_csv('acq_loop_anytime_metrics.csv', index=False)
            # Reset to empty dataframe
            anytime_metrics = pandas.DataFrame(columns= ['Iteration',   'Iteration Mean Obj', 'Mean Obj QD Score', 'Mean Obj QD Score per Bin', 
                                                                        'Iteration Mean Acq', 'Mean Acq QD Score', 'Mean Acq QD Score per Bin',
                                                                        'Percentage Improvements', 'Total Improvements',
                                                                        'Percentage Convergence Errors', 'Convergence Errors',
                                                                        'Total Convergence Errors'])

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
    
    print(f"\n\nMaximize Improvement - New Elites:\n\n")
    
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

        "new_elites"   :   Elites sorted in descending order (new bin discoveries)
        "n_obj_evals"  :   Number of evaluations - allows n_evals != BATCH_SIZE

        "emitter"      :   Used for map_elites if n_elites < n_evals (sampling)
        "target_archive":  Used for map_elites if n_elites < n_evals  (storing)
        "n_map_evals"  :   N evaluations for map_elites
        "target_function"     :   Obj function for map_elites

        "new_elite_archive": (((check if necessary)))
    """
    def ensure_n_samples(improved_elites, new_bin_elites, acq_flag, pred_flag):

        """
        Ensures that the appropiate number of samples is evaluated, by reevaluating Map Elites (if necessary)
        After 4 extra evaluations, the function returns the best elites found so far to avoid infinite loops
        """

        if acq_flag:
            target_archive = self.acq_archive
            n_samples = BATCH_SIZE
        if pred_flag and self.pred_verific_flag:
            target_archive = self.pred_archive
            n_samples = MAX_PRED_VERIFICATION//PREDICTION_VERIFICATIONS

        # If enough elites have been sampled, return
        if n_samples < improved_elites.shape[0] + new_bin_elites.shape[0]:
            return improved_elites, new_bin_elites, n_samples

        # Sample more elites & add improved elites + new bin elites to target_archive
        i_target_archive, i_new_elite_archive = map_elites(self, target_archive=target_archive, target_function=target_function, acq_flag=acq_flag, pred_flag=pred_flag)
        i_improved_elites, i_new_bin_elites = maximize_improvement(i_new_elite_archive, old_elites)
        n_improvements = i_new_elite_archive.stats.num_elites
        iteration = 0

        # Re-enter MAP-Elites (acq/obj) up to 2 times if necessary 
        while n_improvements < n_samples and iteration <= 2:
            print("n_improvements: " + str(n_improvements))
            print("\n\n### Not enough Acq Improvements: Re-entering acquisition loop###\n\n")
            i_target_archive, i_new_elite_archive = map_elites(self, target_archive=i_target_archive, target_function=target_function, new_elite_archive=i_new_elite_archive, acq_flag=acq_flag, pred_flag=pred_flag)
            n_improvements = i_new_elite_archive.stats.num_elites

        # Enough samples have been found, or Loop has been re-entered twice
        # Proceed to sample selection
        i_improved_elites, i_new_bin_elites = maximize_improvement(i_new_elite_archive, old_elites)
        return i_improved_elites, i_new_bin_elites, n_samples


    
    def select_samples(improved_elites, new_bin_elites, n_samples):
        """Selects samples based on exploration behavior defined in the class constructor"""

        if self.explore_flag and not self.greedy_flag:
            # Evaluate new_elites first
            if new_bin_elites.shape[0] >= n_samples:
                candidate_elites = new_bin_elites[:n_samples]
                n_candidate_elites = candidate_elites[:n_samples]
            else:
                candidate_elites = np.concatenate((new_bin_elites, improved_elites), axis=0)
                n_candidate_elites = candidate_elites[:n_samples]

        if self.greedy_flag:
            # Evaluate max_improvement_elites first
            if improved_elites.shape[0] >= n_samples:
                candidate_elites = improved_elites[:n_samples]
                n_candidate_elites = candidate_elites[:n_samples]
            else:
                candidate_elites = np.concatenate((improved_elites, new_bin_elites), axis=0)
                n_candidate_elites = candidate_elites[:n_samples]
        
        return n_candidate_elites



    i_improved_elites, i_new_bin_elites, n_samples = ensure_n_samples(improved_elites=improved_elites, new_bin_elites=new_bin_elites, acq_flag=acq_flag, pred_flag=pred_flag)
    candidate_elites = select_samples(improved_elites=i_improved_elites, new_bin_elites=i_new_bin_elites, n_samples=n_samples)
    eval_xfoil_loop(self, candidate_sol=candidate_elites['solution'], acq_flag=acq_flag, pred_flag=pred_flag)
    # GP Model is updated inside eval_xfoil_loop()


def run_vanilla_sail(self: SailRun):
    """
    Extra evaluations are given if eval_pred_flag is True (see sail.py)
        if not eval_pred_flag, extra_evals = 0
    """

    total_improvements = 0
    total_convergence_errors = 0
    mean_acq_improvement = 0
    eval_budget = ACQ_N_OBJ_EVALS + PRED_N_EVALS
    while(eval_budget >= BATCH_SIZE):
        eval_budget -= BATCH_SIZE

        #acq_archive = store_n_best_elites(acq_archive, obj_archive.stats.num_elites, update_acq=True, gp_model=gp_model, obj_archive=obj_archive)
        acq_archive, _ = map_elites(self, target_archive=self.acq_archive, target_function=acq_ucb, acq_flag=True)                           # evolve acquisition archive

        acq_elite_batch = acq_archive.sample_elites(BATCH_SIZE)        
        acq_elite_solutions = acq_elite_batch[0]
        acq_elite_acquisitions = acq_elite_batch[1]
        acq_elite_measures = acq_elite_batch[2]

        # acq_archive only contains valid solutions (= non-intersecting polynomials), therefore no need to check for validity using valid_indices
        _, surface_area_batch = generate_parsec_coordinates(acq_elite_solutions)
        convergence_errors, success_indices, obj_batch = xfoil(BATCH_SIZE)

        # store evaluations for GP model

        acq_batch = acq_elite_acquisitions[success_indices]
        status_vector, value_vector = self.obj_archive.add(acq_elite_solutions[success_indices], obj_batch, acq_elite_measures[success_indices])
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
        pprint(acq_batch, obj_batch)
        print("Acq Archive Size: " + str(self.acq_archive.stats.num_elites))
        print("Obj Archive Size: " + str(self.obj_archive.stats.num_elites))
        print("Airfoil Convergence Errors: " + str(convergence_errors))
        print("Remaining ACQ Precise Evals: " + str(eval_budget) + "\n\n")

        self.update_gp_model(new_solutions=acq_elite_solutions ,new_objectives=obj_batch)

    return



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

    obj_elite_sol = [elite.solution for elite in self.obj_archive]
    obj_elite_obj = [elite.objective for elite in self.obj_archive]
    obj_elite_bhv = [elite.measures for elite in self.obj_archive]
    self.update_archive(candidate_sol=obj_elite_sol, candidate_obj=obj_elite_obj, candidate_bhv=obj_elite_bhv, pred_flag=True)
    self.update_seed()


    emitter = [
        GaussianEmitter(
        archive=self.pred_archive,
        sigma=SIGMA_EMITTER,
        bounds= np.array(SOL_VALUE_RANGE),
        batch_size=BATCH_SIZE,
        initial_solutions=obj_elite_sol,
        seed=self.current_seed
    )]


    print("\n\n ## Enter Prediction Verification Loop##")

    iteration = 1
    mean_obj = 0
    mean_acq = 0
    i_mean_acq = 0
    i_mean_obj = 0
    total_obj_improvements = 0
    total_acq_improvements = 0
    total_convergence_errors = 0
    mean_acq_improvement = 0
    mean_obj_improvement = 0 # ToDo
    anytime_metrics = pandas.DataFrame(columns=['Iteration', 'Mean Obj', 'Mean Acq', 'Mean Obj Improvement', 'Mean Acq Improvement', 'Percentage Improvements', 'Total Improvements', 'Percentage Convergence Errors', 'Convergence Errors', 'Total Convergence Errors', 'New Acq Elites', 'New Obj Elites'])
    eval_budget = ACQ_N_OBJ_EVALS

    while(eval_budget >= BATCH_SIZE):

        old_acq_elites = np.array([(  elite.solution,     elite.index,     elite.objective,       elite.measures) for elite in self.acq_archive],
                        dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])


        # calculate & evaluate elites, that maximize acquisition values
        # update obj_archive and gp_model inside eval_max_obj_improvement()

        map_elites(self, target_archive=self.acq_archive, target_function=predict_objective, pred_flag=True)
        improved_elites, new_bin_elites = maximize_improvement(new_elite_archive=self.acq_archive, old_elites=old_acq_elites) 
        evaluate_max_improvement(self, improved_elites=improved_elites, new_bin_elites=new_bin_elites, old_elites=old_acq_elites, target_function=predict_objective, acq_flag=True)

        eval_budget -= BATCH_SIZE
        
        # Calculate Anytime Metrics
        # acquisition_improvements = np.vstack(np.hstack((improved_elites['objective_improvement'],new_bin_elites['objective_improvement'])))

        # convergence_errors = self.convergence_errors

        # # Mean Obj & Acq in current iteration
        # i_mean_obj += np.mean(self.new_obj)
        # i_mean_acq += np.mean(self.new_target_elite_obj)
        # mean_obj = np.mean(mean_obj + i_mean_obj)
        # mean_acq = np.mean(mean_acq + i_mean_acq)
        # mean_acq_improvement = np.mean(acquisition_improvements)

        # # Elites in archive before & after
        # obj_t0 = self.obj_t0
        # obj_t1 = self.obj_t1
        # acq_t0 = self.acq_t0
        # acq_t1 = self.acq_t1
        # n_obj_improvements = obj_t1 - obj_t0
        # n_acq_improvements = acq_t1 - acq_t0
        # total_obj_improvements += n_obj_improvements
        # total_acq_improvements += n_acq_improvements
        # percentage_improvements       = (total_obj_improvements/ACQ_N_OBJ_EVALS)*100
        # percentage_convergence_errors = (total_convergence_errors/ACQ_N_OBJ_EVALS)*100

        # # Acq/Obj Metrics
        # qd_obj = self.obj_archive.stats.qd_score
        # qd_acq = self.acq_archive.stats.qd_score
        # print(self.obj_archive.dims)
        # obj_qd_per_bin = qd_obj/self.obj_archive.dims
        # acq_qd_per_bin = qd_acq/self.acq_archive.dims
        # obj_mean_qd_score = qd_obj/self.obj_archive.stats.num_elites
        # acq_mean_qd_score = qd_acq/self.acq_archive.stats.num_elites
        # ToDo: mean improvements (new_converged_elites - old_corresponding_elites) / n_converged_elites 
        # ToDo: print new_converged_elites next to old_corresponding_elites
        # old_corresponding_elites -> right join by index
        # print("Mean Obj Improvement: {:.1f}".format(mean_obj_improvement))
        old_elite_obj = old_acq_elites['objective']

        # Print Anytime Metrics
        # print("Percentage Improvements: {:.1f}".format(percentage_improvements) + "%")
        # print("Percentage Convergence Errors: {:.1f}".format(percentage_convergence_errors) + "%")
        # print("Total Improvements: "       + str(total_obj_improvements))
        # print("Total Convergence Errors: " + str(total_convergence_errors))
        # print("Mean Acq Improvement: {:.1f}".format(mean_acq_improvement))
        # print("   Acq Archive Size (before): " + str(acq_t0))
        # print("   Acq Archive Size  (after): " + str(acq_t1))
        # print("   Mean Acq QD Score: " + str(acq_qd_per_bin))        
        # print("   New Acq Elites: " + str(acq_t1 - acq_t0))
        # print("   Acq QD Score:   " + str(acq_mean_qd_score))
        # print("   Obj QD Score:   " + str(obj_mean_qd_score))
        # print("   New Obj Elites: " + str(obj_t1 - obj_t0))
        # print("   Mean Obj QD Score: " + str(obj_qd_per_bin))
        # print("   Obj Archive Size (before): " + str(obj_t0))
        # print("   Obj Archive Size  (after): " + str(obj_t1))
        # print("Airfoil Convergence Errors: " + str(convergence_errors))
        # print("Remaining ACQ Precise Evals: " + str(eval_budget) + "\n\n")
        # # Store Anytime Metrics in Pandas Dataframe
        # anytime_data = [iteration, i_mean_obj, obj_qd_per_bin, obj_mean_qd_score,
        #                            i_mean_acq, acq_qd_per_bin, acq_mean_qd_score,
        #                            percentage_improvements, total_obj_improvements,
        #                            percentage_convergence_errors, convergence_errors,
        #                            total_convergence_errors]
        # anytime_metrics.loc[len(anytime_metrics)] = anytime_data

        # if iteration % 20 == 0:
        #     # Save Anytime Metrics to CSV. If no csv is saved create new csv, else append to existing csv
        #     try:
        #         anytime_metrics.to_csv('acq_loop_anytime_metrics.csv', mode='a', header=False, index=False)
        #     except:
        #         anytime_metrics.to_csv('acq_loop_anytime_metrics.csv', index=False)
        #     # Reset to empty dataframe
        #     anytime_metrics = pandas.DataFrame(columns= ['Iteration',   'Iteration Mean Obj', 'Mean Obj QD Score', 'Mean Obj QD Score per Bin', 
        #                                                                 'Iteration Mean Acq', 'Mean Acq QD Score', 'Mean Acq QD Score per Bin',
        #                                                                 'Percentage Improvements', 'Total Improvements',
        #                                                                 'Percentage Convergence Errors', 'Convergence Errors',
        #                                                                 'Total Convergence Errors'])

    iteration += 1
    return

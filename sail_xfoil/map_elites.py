###### Import packages #####
from ribs.schedulers import Scheduler
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from tqdm import tqdm
import numpy as np

##### Import custom scripts #####
from utils.utils import calculate_behavior
from utils.pprint_nd import pprint
from xfoil.generate_airfoils import generate_parsec_coordinates
from acq_functions import acq_ucb
from gp import predict_objective

from config.config import Config
config = Config('config/config.ini')
TEST_RUNS = config.TEST_RUNS
BATCH_SIZE = config.BATCH_SIZE
PRED_N_EVALS = config.PRED_N_EVALS
SIGMA_EMITTER = config.SIGMA_EMITTER
SOL_DIMENSION = config.SOL_DIMENSION
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
BHV_NUMBER_BINS = config.BHV_NUMBER_BINS
BHV_VALUE_RANGE = config.BHV_VALUE_RANGE
ACQ_N_MAP_EVALS = config.ACQ_N_MAP_EVALS
PREDICTION_VERIFICATIONS = config.PREDICTION_VERIFICATIONS

def map_elites(self, target_archive, target_function, obj_flag=False, acq_flag=False, pred_flag=False, new_elite_archive=None, pred_verific_flag=False):

    """
    Perform MAP-Elites iterations.

    IMPORTANT: Make sure to update the target archive before entering MAP-Elites
    
    Generates Parsec Coordinates to check if sample is valid.
    Only valid samples are evaluated and added to the archive.
    This makes further considerations of valid indices obsolete.

    Sampled solutions are evaluated on their respective objective
    function. In the case of SAIL, MAP-Elites uses Acquisitions
    & predictions as objective function.

    Newly found elites are added to another archive called
    "new_elites_archive". This archive is used to communicate
    improvements in the target_archive to the calling function.


    The calling function can then leverage information about                # see maximize_improvement()  &  prediction_verification_loop()
    these new elites in order to boost efficiency.

    In order to generalize this function for problem domains
    different than XFOIL, the generate_parsec_coordinates()
    function and the codeblock checking for validity can be
    removed.
    """


    def update_emitter(target_archive, initial_solutions, sigma_emitter=SIGMA_EMITTER, sol_value_range=SOL_VALUE_RANGE):

        emitter = [
            GaussianEmitter(
            archive=target_archive,
            sigma=sigma_emitter,
            bounds= np.array(sol_value_range),
            batch_size=BATCH_SIZE,
            initial_solutions=initial_solutions[:BATCH_SIZE],
            seed=self.current_seed
        )]

        return emitter
    
    print("\n\nInitialize Map-Elites [...]")

    if new_elite_archive is None:
        new_elite_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = -1,)
        
    # initializing archive
    if not acq_flag and not pred_flag:
        initial_solutions = self.init_samples
        n_evals = ACQ_N_MAP_EVALS

    if acq_flag:
        n_evals = ACQ_N_MAP_EVALS
        initial_solutions = np.vstack(([elite.solution for elite in self.obj_archive], [elite.solution for elite in self.acq_archive])) if self.acq_archive.stats.num_elites != 0 else np.array([elite.solution for elite in self.obj_archive])

    if pred_flag:
        initial_solutions = np.vstack(([elite.solution for elite in self.obj_archive], [elite.solution for elite in self.pred_archive])) if self.pred_archive.stats.num_elites != 0 else np.array([elite.solution for elite in self.obj_archive])
        if pred_verific_flag:
            n_evals = PRED_N_EVALS//PREDICTION_VERIFICATIONS
        else:
            n_evals = PREDICTION_VERIFICATIONS

    remaining_evals = n_evals
    total_iterations = remaining_evals // BATCH_SIZE
    
    emitter = update_emitter(target_archive=target_archive, initial_solutions=initial_solutions)
    scheduler = Scheduler(target_archive, emitter)
    
    with tqdm(total=total_iterations) as progress:
        while((remaining_evals-BATCH_SIZE >= 0)):
            self.update_seed()
            progress.update(1)
            valid_indices = np.empty(0, dtype=int) 

            if remaining_evals % 100 == 0:
                update_emitter(target_archive=target_archive, initial_solutions=initial_solutions)
                scheduler = Scheduler(target_archive, emitter)

            # Create Samples
            samples = scheduler.ask()

            if samples.shape[0] != BATCH_SIZE:
                ValueError("Scheduler did not return BATCH_SIZE samples")

            valid_indices, surface_batch = generate_parsec_coordinates(samples)

            scheduler_bhv = samples[:,1:3]  # ToDO: generalize calculate_behavior()
            candidate_sol = samples[valid_indices]
            candidate_obj = target_function(candidate_sol, self.gp_model)
            candidate_bhv = scheduler_bhv[valid_indices]

            # Add Elites to archive, then communicate new archive to SAIL Runner
            self.update_archive(candidate_sol=candidate_sol, candidate_obj=candidate_obj, candidate_bhv=candidate_bhv, obj_flag=obj_flag, acq_flag=acq_flag, pred_flag=pred_flag, surpress_print=True)

            new_sol = self.new_target_elite_sol
            new_obj = self.new_target_elite_obj
            new_bhv = self.new_target_elite_bhv
            new_elite_archive.add(new_sol, new_obj, new_bhv) 
            # store new elites to return to calling function
            # allows to maximize improvement of target_function


            # Scheduler.ask() returns BATCH_SIZE samples
            # Scheduler.tell() expects BATCH_SIZE objectives
            # Insert -1000 for invalid samples to avoid them being selected as elites
            if candidate_obj.shape[0] == BATCH_SIZE:
                scheduler_obj = candidate_obj
            else:
                # create an array equivalent to candidate_obj, but with -1000 for invalid samples
                scheduler_obj = np.full(samples.shape[0], -1000, dtype=float)
                scheduler_obj[valid_indices] = candidate_obj

            scheduler.tell(scheduler_obj, scheduler_bhv)
            remaining_evals -= BATCH_SIZE

    self.update_seed()

    if obj_flag: # Allows benchmarking against default MAP-Elites

        self.update_gp_model(candidate_sol, candidate_obj)

    if acq_flag:
        print(self.acq_archive.stats.num_elites)
    if pred_flag:
        print(self.pred_archive.stats.num_elites)

    return target_archive, new_elite_archive
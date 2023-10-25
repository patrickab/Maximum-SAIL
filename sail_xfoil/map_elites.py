###### Import packages #####
from ribs.schedulers import Scheduler
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from tqdm import tqdm
import subprocess
import numpy as np

##### Import custom scripts #####
from xfoil.generate_airfoils import generate_parsec_coordinates

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

def map_elites(self, target_function, obj_flag=False, acq_flag=False, pred_flag=False, new_elite_archive=None, pred_verific_flag=False):

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
    
    print("\n\nInitialize Map-Elites [...]")

    if new_elite_archive is None:
        new_elite_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = -1,)
        
    target_archive, dummy_solutions, n_evals = define_mapping_behavior(self, acq_flag, pred_flag, pred_verific_flag, target_function)

    size_t0 = target_archive.stats.num_elites
    subprocess.run("rm *.dat", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

    remaining_evals = n_evals
    total_iterations = remaining_evals // BATCH_SIZE

    print("Start Map-Elites [...]")
    obj_t0 = self.obj_archive.stats.num_elites
    print("Acq Elite Archive Size: ", str(self.acq_archive.stats.num_elites))
        
    with tqdm(total=total_iterations) as progress:
        while((remaining_evals-BATCH_SIZE >= 0)):

            progress.update(1)
            valid_indices = np.empty(0, dtype=int) 

            emitter = update_emitter(self, target_archive=target_archive, initial_solutions=dummy_solutions)
            scheduler = Scheduler(target_archive, emitter)

            # Create Samples
            samples = scheduler.ask()

            valid_indices, surface_batch = generate_parsec_coordinates(samples, io_flag=False)
            
            scheduler_bhv = samples[:,1:3]  # ToDO: generalize calculate_behavior()
            candidate_sol = samples[valid_indices]
            candidate_obj = target_function(candidate_sol, self.gp_model)
            candidate_bhv = scheduler_bhv[valid_indices]

            status_vector, _ = target_archive.add(candidate_sol, candidate_obj, candidate_bhv)

            # store newly discovered elites
            non_0_status_indices = np.where(status_vector != 0)[0]            
            new_sol = candidate_sol[non_0_status_indices]
            new_obj = candidate_obj[non_0_status_indices]
            new_bhv = candidate_bhv[non_0_status_indices]
            new_elite_archive.add(new_sol, new_obj, new_bhv)

            # Scheduler.ask() returns BATCH_SIZE samples --- Scheduler.tell() expects BATCH_SIZE objectives 
            if candidate_obj.shape[0] == samples.shape[0]:
                scheduler_obj = candidate_obj
            else:
                # Insert -1000 for invalid samples to avoid them being selected as elites
                scheduler_obj = np.full(samples.shape[0], -1000, dtype=float)
                scheduler_obj[valid_indices] = candidate_obj

            scheduler.tell(scheduler_obj, scheduler_bhv)
            remaining_evals -= BATCH_SIZE

    size_t1 = target_archive.stats.num_elites
    obj_t1 = self.obj_archive.stats.num_elites
    if obj_t1 != obj_t0:
        raise ValueError("MAP-Elites:  obj_t1 != obj_t0   -   debug this!")

    print("Acq Elite Archive Size: ", str(self.acq_archive.stats.num_elites))
    print("End Map-Elites [...]\n\n")

    return target_archive, new_elite_archive, size_t0, size_t1


def define_mapping_behavior(self, acq_flag, pred_flag, pred_verific_flag, target_function):
    """
    Custom version: Use obj elites + acq elites as starting population
    Default version: Use obj elites as starting population
    """

    dummy_elites = sorted(self.obj_archive, key=lambda x: x.objective, reverse=True)[:BATCH_SIZE]
    dummy_solutions = np.array([elite.solution for elite in dummy_elites])

    obj_df = self.obj_archive.as_pandas()
    obj_elites_sol = obj_df.values[:,4:]
    obj_elites_bhv = obj_df.values[:,1:3]
    obj_elites_obj = target_function(obj_elites_sol, self.gp_model)

    if acq_flag:
        n_evals = ACQ_N_MAP_EVALS

        if self.custom_flag:
            target_archive = self.acq_archive
        if self.vanilla_flag:
            self.acq_archive.clear()
            self.acq_archive.add(obj_elites_sol, obj_elites_obj, obj_elites_bhv)
            target_archive = self.acq_archive

    if pred_flag:

        if self.pred_verific_flag:
            n_evals = PRED_N_EVALS//(PREDICTION_VERIFICATIONS+1)
        else:
            n_evals = PRED_N_EVALS

        if self.custom_flag:
            target_archive = self.pred_archive
        if self.vanilla_flag:
            self.pred_archive.clear()
            self.pred_archive.add(obj_elites_sol, obj_elites_obj, obj_elites_bhv)
            target_archive = self.pred_archive
            
    return target_archive, dummy_solutions, n_evals


def update_emitter(self, target_archive, initial_solutions, sigma_emitter=SIGMA_EMITTER, sol_value_range=SOL_VALUE_RANGE):

    self.update_seed()

    emitter = [
        GaussianEmitter(
        archive=target_archive,
        sigma=sigma_emitter,
        bounds= np.array(sol_value_range),
        batch_size=BATCH_SIZE,
        initial_solutions=initial_solutions,
        seed=self.current_seed
    )]

    return emitter

###### Import packages #####
from ribs.schedulers import Scheduler
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.emitters._emitter_base import EmitterBase
from tqdm import tqdm
import subprocess
import numpy as np

##### Import custom scripts #####
from xfoil.generate_airfoils import generate_parsec_coordinates
from gp.predict_objective import predict_objective

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

def map_elites(self, acq_flag=False, pred_flag=False, new_elite_archive=None, pred_verific_flag=False):

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

    subprocess.run("rm *.dat", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        
    if acq_flag:
        target = "Acq Archive"
        target_function = self.acq_function
        target_archive = self.acq_archive
        if self.vanilla_flag:
            obj_df = self.obj_archive.as_pandas(include_solutions=True)
            self.acq_archive.clear()
            self.acq_archive.add(obj_df.solution_batch(), obj_df.objective_batch(), obj_df.measures_batch())
        size_t0 = self.acq_archive.stats.num_elites
        n_evals = ACQ_N_MAP_EVALS # reduce number of acquisition evaluations for MES
    if pred_flag:
        target = "Pred Archive"
        target_function = predict_objective
        target_archive = self.pred_archive
        if self.vanilla_flag:
            obj_df = self.obj_archive.as_pandas(include_solutions=True)
            self.pred_archive.clear()
            self.pred_archive.add(obj_df.solution_batch(), obj_df.objective_batch(), obj_df.measures_batch())
        size_t0 = self.pred_archive.stats.num_elites
        n_evals = PRED_N_EVALS if not self.pred_verific_flag else PRED_N_EVALS//(PREDICTION_VERIFICATIONS+1)

    remaining_evals = n_evals
    total_iterations = remaining_evals // BATCH_SIZE
    obj_t0 = self.obj_archive.stats.num_elites
    print(f"{target} Size: ", str(size_t0))
        
    with tqdm(total=total_iterations) as progress:
        while((remaining_evals-BATCH_SIZE >= 0)):

            progress.update(1)
            valid_indices = np.empty(0, dtype=int) 

            emitter = update_emitter(self, target_archive=target_archive)
            scheduler = Scheduler(target_archive, emitter)

            # Create Samples
            samples = scheduler.ask()

            # Generate Parsec Coordinates & remove Invalid Samples
            valid_indices, surface_batch = generate_parsec_coordinates(samples, io_flag=False)
            
            scheduler_bhv = samples[:,1:3]  # ToDO: generalize calculate_behavior()
            candidate_sol = samples[valid_indices]
            candidate_obj = target_function(candidate_sol, self.gp_model)
            candidate_bhv = scheduler_bhv[valid_indices]

            status_vector, _ = target_archive.add(solution_batch=candidate_sol, objective_batch=candidate_obj, measures_batch=candidate_bhv)
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

    obj_t1 = self.obj_archive.stats.num_elites
    size_t1 = self.acq_archive.stats.num_elites if acq_flag else self.pred_archive.stats.num_elites
    target_size_t1 = target_archive.stats.num_elites

    print(f"{target} Size: ", str(size_t1))
    if acq_flag and pred_flag:
        raise ValueError("MAP-Elites:  acq_flag and pred_flag both True   -   debug this!")
    if obj_t0 != obj_t1:
        raise ValueError("MAP-Elites:  obj_t0 != obj_t1   -   debug this!")
    if size_t0 > size_t1:
        raise ValueError("MAP-Elites:  size_t0 < size_t1   -   debug this!")
    if target_size_t1 != size_t1:
        raise ValueError("MAP-Elites:  target_size_t1 != size_t1   -   debug this!")


    print("[...] End Map-Elites\n\n")

    return new_elite_archive, size_t0, size_t1


def update_emitter(self, target_archive, sigma_emitter=SIGMA_EMITTER, sol_value_range=SOL_VALUE_RANGE):

    self.update_seed()

    emitter = [
        ScaledGaussianEmitter(
        archive=target_archive,
        sigma=sigma_emitter,
        bounds= np.array(sol_value_range),
        batch_size=BATCH_SIZE,
        seed=self.current_seed
    )]

    return emitter


class ScaledGaussianEmitter(GaussianEmitter):

    """
    Custom Emitter class
        - Adds Gaussian Noise scaled to solution space boundaries
        - Requires filled archive instead of initial solutions
    """

    def __init__(self,
                 archive,
                 *,
                 sigma,
                 bounds=None,
                 batch_size=BATCH_SIZE,
                 seed=None):

        self._rng = np.random.default_rng(seed)
        self._batch_size = batch_size
        self._sigma = np.array(sigma, dtype=archive.dtype)

        if archive.stats.num_elites == 0:
            raise ValueError("Archive must be filled with initial solutions.")

        EmitterBase.__init__(
            self,
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )

    @property
    def sigma(self):
        """float or numpy.ndarray: Standard deviation of the (diagonal) Gaussian
        distribution when the archive is not empty."""
        return self._sigma

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    def ask(self):
        """Creates solutions by adding Gaussian noise to elites in the archive.

        Each solution is drawn from a distribution centered at a randomly
        chosen elite with standard deviation ``self.sigma``.
        """

        parents = self.archive.sample_elites(self._batch_size).solution_batch

        scaled_noise = self._rng.normal(
            scale=np.abs(self._sigma*(self.upper_bounds-self.lower_bounds)),
            size=(self._batch_size, self.solution_dim),
        )

        return np.clip(parents + scaled_noise, self.lower_bounds, self.upper_bounds)

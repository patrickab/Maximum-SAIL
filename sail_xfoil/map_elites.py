###### Import packages #####
from ribs.schedulers import Scheduler
from ribs.archives import GridArchive
from tqdm import tqdm
import numpy
import torch
import gc

##### Import custom scripts #####
from utils.pprint_nd import pprint_nd, pprint, pprint_fstring
from xfoil.generate_airfoils import generate_parsec_coordinates

from config.config import Config
config = Config('config/config.ini')
TEST_RUNS = config.TEST_RUNS
BATCH_SIZE = config.BATCH_SIZE
SOL_DIMENSION = config.SOL_DIMENSION
BHV_NUMBER_BINS = config.BHV_NUMBER_BINS
BHV_VALUE_RANGE = config.BHV_VALUE_RANGE

def map_elites(self, target_archive, emitter, n_evals, fuct_obj, obj_flag=False, acq_flag=False, pred_flag=False, new_elite_archive=None):
    
    print("\n\nInitialize Map-Elites [...]")

    if new_elite_archive is None:
        new_elite_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = -1,)

    scheduler = Scheduler(target_archive, emitter)

    remaining_evals = n_evals
    total_iterations = remaining_evals // BATCH_SIZE
    
    with tqdm(total=total_iterations) as progress:
        while((remaining_evals-BATCH_SIZE >= 0)):

            progress.update(1)
            valid_indices = numpy.empty(0, dtype=int) 

            samples = scheduler.ask() 
            self.update_seed()

            valid_indices, surface_batch = generate_parsec_coordinates(samples)

            scheduler_bhv = samples[:,[1,2]]

            candidate_sol = samples[valid_indices]
            candidate_obj = fuct_obj(candidate_sol, self.gp_model)
            candidate_bhv = scheduler_bhv[valid_indices]

            self.update_archive(candidate_sol, candidate_obj, candidate_bhv, obj_flag, acq_flag, pred_flag)
            status_vector, _ = target_archive.add(candidate_sol, candidate_obj, candidate_bhv)
                
            non_0_status_indices = numpy.where(status_vector != 0)[0]

            new_sol = candidate_sol[non_0_status_indices]
            new_obj = candidate_obj[non_0_status_indices]
            new_bhv = candidate_bhv[non_0_status_indices]

            new_elite_archive.add(new_sol, new_obj, new_bhv)

            # Insert -1000 for invalid samples to avoid them being selected as elites
            for index in range(samples.shape[0]):
                if index not in valid_indices:
                    candidate_obj = numpy.insert(candidate_obj, index, -1000, axis=0)
                    numpy.insert

            scheduler.tell(candidate_obj, scheduler_bhv)
            remaining_evals -= BATCH_SIZE

    if obj_flag:
        self.update_gp_model(candidate_sol, candidate_obj)
    return target_archive, new_elite_archive
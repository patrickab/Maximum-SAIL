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

def map_elites(archive, emitter, gp_model, n_evals, fuct_obj):
    
    print("\nInitialize Map-Elites [...]")

    new_elite_archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = -1,
    )
    
    new_elite_archive.clear()

    scheduler = Scheduler(archive, emitter)

    remaining_evals = n_evals
    total_iterations = remaining_evals // BATCH_SIZE
    
    with tqdm(total=total_iterations) as progress:
        while(remaining_evals-BATCH_SIZE >= 0):
            progress.update(1)

            valid_indices = numpy.empty(0, dtype=int) 

            samples = scheduler.ask()
            archive._seed += TEST_RUNS

            valid_indices, surface_area_batch = generate_parsec_coordinates(samples)
            valid_samples = samples[valid_indices]

            bhv_evals = samples[:,[1,2]]
            obj_evals = fuct_obj(valid_samples, gp_model)

            # status_batch[0] = not added       status_batch[1] = added         status_batch[2] = updated
            status_batch, value_batch = archive.add(valid_samples, obj_evals, bhv_evals[valid_indices])
            non_0_status_indices = numpy.where(status_batch != 0)[0]
            new_elites = valid_samples[non_0_status_indices]

            new_elite_archive.add(new_elites, obj_evals[non_0_status_indices], bhv_evals[valid_indices][non_0_status_indices])

            # Insert -1000 for invalid samples to avoid them being selected as elites
            for index in range(samples.shape[0]):
                if index not in valid_indices:
                    obj_evals = numpy.insert(obj_evals, index, -1000, axis=0)

            scheduler.tell(obj_evals, bhv_evals)
            remaining_evals -= BATCH_SIZE

    #print("Elites in Archive: " + str(archive.stats.num_elites))
    print("[...] Terminate Map-Elites\n")

    print("Elites in New Elite Archive: " + str(new_elite_archive.stats.num_elites))
    
    return archive, new_elite_archive


def select_new_elites(elite_status_vector,candidate_elites, obj_evals):

    x_new_elites = []
    obj_new_elites = []

    for candidate_elite, obj_eval ,elite_status_value in candidate_elites, obj_evals, elite_status_vector:
        if elite_status_value > 0:
            x_new_elites.append(candidate_elite)
            obj_new_elites.append(obj_eval)

    return x_new_elites, obj_new_elites
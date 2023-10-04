###### Import packages #####
from ribs.schedulers import Scheduler
from tqdm import tqdm
import numpy
import torch

##### Import custom scripts #####
from utils.pprint_nd import pprint_nd, pprint
from xfoil.generate_airfoils import generate_parsec_coordinates

from config.config import Config
config = Config('config/config.ini')
TEST_RUNS = config.TEST_RUNS
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
BATCH_SIZE = config.BATCH_SIZE

def map_elites(archive, emitter, gp_model, n_evals, fuct_obj):
    
    print("\nInitialize Map-Elites [...]")

    # Archive Scheduler
    scheduler = Scheduler(archive, emitter)

    remaining_evals = n_evals
    total_iterations = remaining_evals // BATCH_SIZE
    
    with tqdm(total=total_iterations) as progress:
        while(remaining_evals-BATCH_SIZE >= 0):
            progress.update(1)

            valid_indices = numpy.empty(0, dtype=int) 

            samples = scheduler.ask()
            archive._seed += TEST_RUNS
            #print("Seed: " + str(archive._seed))

            valid_indices = generate_parsec_coordinates(samples)
            valid_samples = samples[valid_indices]

            bhv_evals = samples[:,[1,2]]
            obj_evals = fuct_obj(valid_samples, gp_model)

            # Insert -1000 for invalid samples to avoid them being selected as elites
            for index in range(samples.shape[0]):
                if index not in valid_indices:
                    obj_evals = numpy.insert(obj_evals, index, -1000, axis=0)

            scheduler.tell(obj_evals, bhv_evals)
            remaining_evals -= BATCH_SIZE

    print("Elites in Archive: " + str(archive.stats.num_elites))
    print("[...] Terminate Map-Elites\n")
    
    return archive
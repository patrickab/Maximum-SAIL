###### Import packages #####
from ribs.schedulers import Scheduler
import numpy

##### Import custom scripts #####
from utils.pprint import pprint

from config import Config
config = Config('config.ini')
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
PARALLEL_BATCH_SIZE = config.PARALLEL_BATCH_SIZE

def map_elites(archive, emitter, gp_model, n_evals, fuct_obj, fuct_bhv, fuct_variation_operator=None):
    
    print("\nInitialize Map-Elites [...]\n")

    # Archive Scheduler
    scheduler = Scheduler(archive, emitter)

    remaining_evals = n_evals
    while((remaining_evals)>0):    
        
        print("Enter MAP Loop")
        print("Remaining Evals: " + str(remaining_evals))
        samples = scheduler.ask()

        # Variation Operator
        obj_evals = fuct_obj(samples, gp_model)
        bhv_evals = fuct_bhv(samples)

        remaining_evals -= PARALLEL_BATCH_SIZE

        print("Elites in Archive (before): " + str(archive.stats.num_elites))
        elite_status_vector = scheduler.tell(obj_evals, bhv_evals)
        print("Elites in Archive  (after): " + str(archive.stats.num_elites))


        print("Exit MAP Loop")
        if remaining_evals > 0:
            print()


    print("\n[...] Terminate Map-Elites\n")
    
    return archive
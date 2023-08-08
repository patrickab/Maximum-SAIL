###### Import packages #####
from ribs.schedulers import Scheduler
import numpy

##### Import custom scripts #####
from utils.pprint import pprint

from config import Config
config = Config('config.ini')
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
PARALLEL_BATCH_SIZE = config.PARALLEL_BATCH_SIZE

def map_elites(archive, emitter, gp_model, n_evals, fuct_objective, fuct_behavior, fuct_variation_operator):
    
    print("\nInitialize Map-Elites [...]\n")

    # Archive Scheduler
    scheduler = Scheduler(archive, emitter)

    remaining_evals = n_evals
    while((remaining_evals)>0):    
        
        print("Enter Loop")
        print("Remaining Evals: " + str(remaining_evals))
        sol_candidates = scheduler.ask() # Generates n=PARALLEL_BATCH_SIZE solutions according to emitter

        # Variation Operator

        obj_evals = fuct_objective(sol_candidates, gp_model)          # Calculate objective
        bhv_evals = fuct_behavior(sol_candidates)                     # Calculate behavior

        # for i in range(len(sol_candidates)):
        #     sol_candidate = sol_candidates[i]
        #     for j in range(len(sol_candidate)):
        #         lower_bound, upper_bound = SOL_VALUE_RANGE[j]
        #         sol_candidates[i][j] = (sol_candidates[i][j] % upper_bound+1) + lower_bound

        remaining_evals -= PARALLEL_BATCH_SIZE

        print("Elites in Archive (before): " + str(archive.stats.num_elites))
        elite_status_vector = scheduler.tell(obj_evals, bhv_evals)
        print("Elites in Archive  (after): " + str(archive.stats.num_elites))


        print("Exit Loop")
        if remaining_evals > 0:
            print()


    print("\n[...] Terminate Map-Elites\n")
    
    return archive
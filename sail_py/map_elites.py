###### Import packages #####
from ribs.schedulers import Scheduler
import numpy

##### Import custom scripts #####
from example.example_functions import example_objective_function
from utils.pprint import pprint

from config import Config
config = Config('config.ini')
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
PARALLEL_BATCH_SIZE = config.PARALLEL_BATCH_SIZE

def map_elites(archive, emitter, n_evals, fuct_objective, fuct_behavior, fuct_variation_opeartor):
    
    print("\nInitialize Map-Elites [...]\n")

    # Archive Scheduler
    scheduler = Scheduler(archive, emitter)

    remaining_evals = n_evals
    while((remaining_evals)>0):    
        
        print("Enter Loop")
        print("Remaining Evals: " + str(remaining_evals))
        sol_candidates = scheduler.ask() # Generates n=PARALLEL_BATCH_SIZE solutions according to emitter

        # Variation Operator

        obj_evals = example_objective_function(sol_candidates)      # Calculate objective       (replace with dynamic 'fuct_objective' parameters)
        bhv_evals = fuct_behavior(sol_candidates)                   # Calculate behavior

        for i in range(len(sol_candidates)):
            sol_candidate = sol_candidates[i]
            for j in range(len(sol_candidate)):
                lower_bound, upper_bound = SOL_VALUE_RANGE[j]
                sol_candidates[i][j] = (sol_candidates[i][j] % upper_bound+1) + lower_bound

        remaining_evals -= 250 # after debugging: set back to PARRALLEL_BATCH_SIZE

        print("Update Archive")

        elite_status_vector = scheduler.tell(obj_evals, bhv_evals)
        print("Exit Loop")
        if remaining_evals > 0:
            print()


    print("\n[...] Terminate Map-Elites\n")
    
    return archive
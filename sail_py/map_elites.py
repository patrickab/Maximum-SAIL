###### Archive packages #####
from ribs.schedulers import Scheduler

from config import Config
config = Config('config.ini')
PARALLEL_BATCH_SIZE = config.PARALLEL_BATCH_SIZE
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
EMITTER = config.EMITTER
ARCHIVE = config.ARCHIVE

def map_elites(n_evals, fuct_objective, fuct_behavior, fuct_variation_opeartor):
        
    global ARCHIVE
    
    # Archive Scheduler
    scheduler = Scheduler(ARCHIVE, EMITTER)

    for i in n_evals:

        # Generate n=PARALLEL_BATCH_SIZE candidate solutions according to emitter
        sol_candidates = scheduler.ask()

        # Apply variation operator
        sol_candidates = fuct_variation_opeartor(sol_candidates)

        # Evaluate Performance and Behavior
        obj_evals = fuct_objective(sol_candidates)
        bhv_evals = fuct_behavior(sol_candidates)

        i += PARALLEL_BATCH_SIZE

        # Update Archive
        scheduler.tell(obj_evals, bhv_evals)
    
    return ARCHIVE
###### Archive packages #####
from ribs.emitters import GaussianEmitter
from ribs.schedulers import Scheduler

PARALLEL_BATCH_SIZE = 10
SOL_VALUE_RANGE = [
    (0,10), # dim1
    (5,10), # dim2
    (2,32)  # dim3
    ]

def map_elites(n_evals, sol_archive, fuct_objective, fuct_behavior, fuct_variation_opeartor):
        
    # Method for generating new offspring
    emitters = [
        GaussianEmitter(
            sol_archive,
            sigma=0.5,
            bounds= SOL_VALUE_RANGE,
            batch_size=PARALLEL_BATCH_SIZE
        )
    ]

    # Archive Scheduler
    scheduler = Scheduler(sol_archive, emitters)

    for i in n_evals:

        # Generate n=PARALLEL_BATCH_SIZE candidate solutions according to emitter
        sol_candidates = scheduler.ask()

        # Apply variation operator
        sol_candidates = fuct_variation_opeartor(sol_candidates)

        # Evaluate Performance and Behavior
        obj_evals = fuct_objective(sol_candidates)
        bhv_evals = fuct_behavior(sol_candidates)

        gp_data = (sol_candidates,obj_evals)

        i += PARALLEL_BATCH_SIZE

        # Update Archive
        scheduler.tell(obj_evals, bhv_evals)
    
    return sol_archive, gp_data


from ..config import Config
config = Config('../config.ini')
SOL_DIMENSION = config.SOL_DIMENSION
INIT_ARCHIVE_SIZE = config.INIT_ARCHIVE_SIZE
SOL_DISTRIBUTION = config.SOL_DISTRIBUTION
ARCHIVE = config.ARCHIVE

# logic allows (archiv.size() > INIT_ARCHIVE_SIZE) but ensures (archiv.size() >= INIT_ARCHIVE_SIZE)
def initialize_archive(example_objective_function, example_behavior_function):

    global ARCHIVE

    while ARCHIVE.size() < INIT_ARCHIVE_SIZE:
        init_solutions = SOL_DISTRIBUTION.sample(                    # Generate initial solutions
            INIT_ARCHIVE_SIZE,
            rule="sobol")
    
        init_obj_evals = example_objective_function(init_solutions)       # Calculate objective
        bhv_evals = example_behavior_function(init_solutions)             # Calculate behavior 
        ARCHIVE.add(init_solutions, init_obj_evals, bhv_evals)            # Store elite solutions

    return ARCHIVE, init_solutions, init_obj_evals
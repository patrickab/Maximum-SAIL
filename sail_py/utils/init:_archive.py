INIT_ARCHIVE_SIZE = 50      # Note: current state of code 
                            # allows (archiv.size() > INIT_ARCHIVE_SIZE) 
                            # but ensures (archiv.size() >= INIT_ARCHIVE_SIZE)


    # logic allows (archiv.size() > INIT_ARCHIVE_SIZE) but ensures (archiv.size() >= INIT_ARCHIVE_SIZE)
def initialize_archive(archive, sol_distribution, INIT_ARCHIVE_SIZE, example_objective_function, example_behavior_function):
    while archive.size() < INIT_ARCHIVE_SIZE:
        init_solutions = sol_distribution.sample(                    # Generate initial solutions
            INIT_ARCHIVE_SIZE,
            rule="sobol")
    
        obj_evals = example_objective_function(init_solutions)       # Calculate objective
        bhv_evals = example_behavior_function(init_solutions)        # Calculate performance 
        archive.add(init_solutions, obj_evals, bhv_evals)            # Save elite solutions

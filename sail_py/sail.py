###### Archive packages #####
from ribs.archives import GridArchive
from chaospy import Uniform, Iid
import numpy as np

###### Import Custom Scripts ######
import acq_normal_distribution
import map_elites
import train_GP


######## Define Parameters ######## 
# (adjust settings in map_elites.py)
EVALUATION_BUDGET = 1000
ACQ_N_EVALS = 1000
PRED_N_EVALS = 1000
PARALLEL_BATCH_SIZE = 10
# Define solution space
SOL_DIMENSION = [3]
SOL_VALUE_RANGE = [
    (0,10), # dim1
    (5,10), # dim2
    (2,32)  # dim3
    ]

INIT_ARCHIVE_SIZE = 50      # Note: current state of code 
                            # allows (archiv.size() > INIT_ARCHIVE_SIZE) 
                            # but ensures (archiv.size() >= INIT_ARCHIVE_SIZE)



######### Define Archive #########ä
# Note: current state of work does not differentiate between 
# behavior in acquisition vs prediction map
BHV_ARCHIVE_DIMENSION = [2]
BHV_NUMBER_BINS = [50,20]
BHV_VALUE_RANGE = [
    (0,100), # dim1
    (10,20)  # dim2
    ]



def example_objective_function(genome):
    
    genome_sum = 0
    for dim in genome:
        genome_sum += dim

    return genome_sum
def example_behavior_function(genome):

    bhv_vector = [None,None]
    
    # nothing meaningful being done, just calculates "random" behavioral values within value range from given genome input
    bhv_vector[0] = 0
    for dim in genome:
        bhv_vector[0] += dim*3.1415 % 100 # generates value between 0 and 100

    bhv_vector[1] = 0
    for dim in genome:
        bhv_vector[1] += dim*2.7182 % 20 # generates value between 0 and 20
def example_variation_function(genomes):
    
    for genome in genomes:
        i=0
        for dim in genome:
            if i%2 == 0:
                dim += -1
            else:
                dim -=  1
            i += 1


example_archive = GridArchive(
    solution_dim=SOL_DIMENSION,         # Dimension of solution vector
    dims=BHV_ARCHIVE_DIMENSION,         # Dimension of behavior vector
    ranges=BHV_VALUE_RANGE,             # Possible values for behavior vector
    qd_score_offset=-600
)

sol_distribution = Iid([                # Define uniform iid distribution among dimensions
    Uniform(lower_bound, upper_bound)
        for lower_bound, upper_bound in SOL_VALUE_RANGE
])

def predict_objective(gp_model): 
    return np.mean(gp_model)

def sail(archive):

    # logic allows (archiv.size() > INIT_ARCHIVE_SIZE) but ensures (archiv.size() >= INIT_ARCHIVE_SIZE)
    while archive.size() < INIT_ARCHIVE_SIZE:
        init_solutions = sol_distribution.sample(                    # Generate initial solutions
            INIT_ARCHIVE_SIZE,
            rule="sobol")
    
        obj_evals = example_objective_function(init_solutions)       # Calculate objective
        bhv_evals = example_behavior_function(init_solutions)        # Calculate performance
        ACQ_N_EVALS -= PARALLEL_BATCH_SIZE 
        archive.add(init_solutions, obj_evals, bhv_evals)            # Save elite solutions

    gp_observations = (init_solutions, obj_evals)
    # (... train GP)


    ########################
    ### Acquisition loop ###
    ########################

    # (...do things)
    n_evals = ACQ_N_EVALS
    while(n_evals-PARALLEL_BATCH_SIZE >= 0): # future addition: add threshhold condition for predictive model performance
        archive, gp_observations = map_elites(PARALLEL_BATCH_SIZE, archive, acq_normal_distribution(), example_behavior_function(), example_variation_function())
        n_evals -= PARALLEL_BATCH_SIZE
        train_GP(gp_observations)
        # future addition: calculate residuals & build mean over predictive model performance

    # (...do things)

    ########################
    ### Prediction loop ####
    ########################

    # (... do things)
    n_evals = PRED_N_EVALS
    while(n_evals-PARALLEL_BATCH_SIZE >= 0):
        archive, gp_observations = map_elites(PARALLEL_BATCH_SIZE, archive, predict_objective(), example_behavior_function(), example_variation_function())
        n_evals-=PARALLEL_BATCH_SIZE
    # (...do things)
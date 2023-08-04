###### Archive packages #####
from ribs.archives import GridArchive
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from chaospy import Uniform, Iid
import numpy as np
import torch

###### Import Custom Scripts ######
import acq_functions.acq_normal_distribution as acq_normal_distribution
from train_GP import init_gp_model, update_gp_model
import map_elites
import utils.initialize_archive as initialize_archive

###### Configurable Variables ######
from config import Config
config = Config('config.ini')
ACQ_N_EVALS = config.ACQ_N_EVALS
PRED_N_EVALS = config.PRED_N_EVALS
PARALLEL_BATCH_SIZE = config.PARALLEL_BATCH_SIZE
SOL_DIMENSION = config.SOL_DIMENSION
BHV_ARCHIVE_DIMENSION = config.BHV_ARCHIVE_DIMENSION
BHV_VALUE_RANGE = config.BHV_VALUE_RANGE

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

# Reminder: Architecture of Grid Archive Datatype
example_archive = GridArchive(
    solution_dim=SOL_DIMENSION,         # Dimension of solution vector
    dims=BHV_ARCHIVE_DIMENSION,         # Dimension of behavior vector
    ranges=BHV_VALUE_RANGE,             # Possible values for behavior vector
    qd_score_offset=-600
)

def predict_objective(gp_model): 
    return np.mean(gp_model)

def sail(archive):

    # current logic allows (archiv.size() > INIT_ARCHIVE_SIZE) but ensures (archiv.size() >= INIT_ARCHIVE_SIZE)
    archive, init_solutions, init_obj_evals = initialize_archive(archive, example_objective_function(),example_behavior_function())
    gp_model = init_gp_model(init_solutions, init_obj_evals)
    
    #### ACQUISITION LOOP
    eval_budget = ACQ_N_EVALS

    while(eval_budget-PARALLEL_BATCH_SIZE >= 0): # future addition: add threshhold condition for predictive model performance
        
        # Calculate and store acquisition elites
        archive = map_elites(ACQ_N_EVALS, archive, acq_normal_distribution(), example_behavior_function(), example_variation_function())
        
        # Select & evaluate acquisition elites (sobol sample in original paper)
        acq_elites = archive.sample_elites(PARALLEL_BATCH_SIZE)
        obj_evals = example_objective_function(acq_elites)
        bhv_evals = example_behavior_function(acq_elites)

        obj_eval_archive = obj_eval_archive + (acq_elites, obj_evals)

        eval_budget -= PARALLEL_BATCH_SIZE

        gp_model = update_gp_model(obj_eval_archive)
        # future addition: calculate residuals & build mean over predictive model performance
    
    ##### PREDICTION LOOP
    eval_budget = PRED_N_EVALS

    while(eval_budget-PARALLEL_BATCH_SIZE >= 0):
        archive = map_elites(PARALLEL_BATCH_SIZE, archive, predict_objective(), example_behavior_function(), example_variation_function())
        eval_budget-=PARALLEL_BATCH_SIZE

    return archive
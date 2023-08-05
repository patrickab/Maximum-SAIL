###### Archive packages #####
import numpy as np

###### Import Custom Scripts ######
import map_elites
import acq_functions.acq_normal_distribution as acq_normal_distribution
import utils.initialize_archive as initialize_archive
from train_GP import fit_gp_model, update_gp_model
from example.example_functions import example_objective_function, example_behavior_function, example_variation_function

###### Configurable Variables ######
from config import Config
config = Config('config.ini')
ACQ_N_EVALS = config.ACQ_N_EVALS
PRED_N_EVALS = config.PRED_N_EVALS
PARALLEL_BATCH_SIZE = config.PARALLEL_BATCH_SIZE
SOL_DIMENSION = config.SOL_DIMENSION
BHV_ARCHIVE_DIMENSION = config.BHV_ARCHIVE_DIMENSION
BHV_VALUE_RANGE = config.BHV_VALUE_RANGE
ARCHIVE = config.ARCHIVE

def predict_objective(gp_model): 
    return np.mean(gp_model)

def sail():

    global ARCHIVE

    # current logic allows (archiv.size() > INIT_ARCHIVE_SIZE) but ensures (archiv.size() >= INIT_ARCHIVE_SIZE)
    ARCHIVE, init_solutions, init_obj_evals = initialize_archive(example_objective_function(),example_behavior_function())
    obj_eval_archive = obj_eval_archive + (init_solutions, init_obj_evals)

    gp_model = fit_gp_model(obj_eval_archive)
    
    #### ACQUISITION LOOP
    eval_budget = ACQ_N_EVALS

    while(eval_budget-PARALLEL_BATCH_SIZE >= 0): # future addition: add threshhold condition for predictive performance of the model
        
        # Calculate and store acquisition elites
        ARCHIVE = map_elites(ACQ_N_EVALS, acq_normal_distribution(), example_behavior_function(), example_variation_function())
        
        # Select & evaluate acquisition elites (sobol sample in original paper)
        acq_elites = ARCHIVE.sample_elites(PARALLEL_BATCH_SIZE)
        obj_evals = example_objective_function(acq_elites)

        obj_eval_archive = obj_eval_archive + (acq_elites, obj_evals)
        eval_budget -= PARALLEL_BATCH_SIZE

        gp_model = fit_gp_model(obj_eval_archive)

    #### PREDICTION MAP
    ARCHIVE = map_elites(PRED_N_EVALS, predict_objective(gp_model), example_behavior_function(), example_variation_function())

    return ARCHIVE
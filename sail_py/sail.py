###### Archive packages #####
import numpy as np
from ribs.emitters import GaussianEmitter
from ribs.archives import GridArchive


###### Import Custom Scripts ######
from acq_functions.acq_normal_distribution import acq_normal_distribution
from example.example_functions import example_objective_function, example_behavior_function, example_variation_function
from utils.initialize_archive import initialize_archive
from utils.train_GP import fit_gp_model
from utils.pprint import pprint
from map_elites import map_elites

###### Configurable Variables ######
from config import Config
config = Config('config.ini')
ACQ_N_OBJ_EVALS = config.ACQ_N_OBJ_EVALS
ACQ_N_MAP_EVALS = config.ACQ_N_MAP_EVALS
PRED_N_EVALS = config.PRED_N_EVALS
PARALLEL_BATCH_SIZE = config.PARALLEL_BATCH_SIZE
SOL_DIMENSION = config.SOL_DIMENSION
BHV_ARCHIVE_DIMENSION = config.BHV_ARCHIVE_DIMENSION
BHV_VALUE_RANGE = config.BHV_VALUE_RANGE
BHV_NUMBER_BINS = config.BHV_NUMBER_BINS
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE

def predict_objective(gp_model): 
    return 1

def sail():

    print("Initialize sail() [...]")

    archive = GridArchive(
        solution_dim=SOL_DIMENSION,         # Dimension of solution vector
        dims=BHV_NUMBER_BINS,               # Dimension of behavior vector
        ranges=BHV_VALUE_RANGE,             # Possible values for behavior vector
        qd_score_offset=-600)
    
    archive, init_solutions, init_obj_evals = initialize_archive(archive, example_objective_function,example_behavior_function)

    pprint(init_solutions)
    print()
    pprint(init_obj_evals)

    sol_archive = []
    obj_archive = []

    sol_archive = init_solutions
    obj_archive = init_obj_evals

    gp_model = fit_gp_model(sol_archive, obj_archive)

    #### ACQUISITION LOOP

    emitter = [
        GaussianEmitter(
        archive=archive,
        sigma=0.5,
        bounds= SOL_VALUE_RANGE,
        batch_size=PARALLEL_BATCH_SIZE,
        initial_solutions=init_solutions
    )]

    eval_budget = ACQ_N_OBJ_EVALS
    print("\n Enter Acquisition Loop")
    while(eval_budget-PARALLEL_BATCH_SIZE >= 0): # future addition: add threshhold condition for predictive performance of the model
        
        # Calculate and store acquisition elites
        archive = map_elites(archive, emitter, ACQ_N_MAP_EVALS, acq_normal_distribution, example_behavior_function, example_variation_function)
        
        # Select & evaluate acquisition elites (sobol sample in original paper)
        acq_elites = archive.sample_elites(PARALLEL_BATCH_SIZE)
        obj_evals = example_objective_function(acq_elites[0])
        # Select/Restructure returned values
        acq_elites = acq_elites[0]        
        obj_evals = obj_evals.reshape(-1,1)

        sol_archive = np.vstack((sol_archive, acq_elites), dtype=float)
        print("\n\n")
        pprint(obj_archive)
        print()
        pprint(obj_evals)

        obj_archive = np.vstack((obj_archive, obj_evals), dtype=float)

        print("appended new solutions to archives")

        if eval_budget == ACQ_N_OBJ_EVALS:
            observation_archive = (acq_elites, obj_evals)
        else:
            observation_archive = observation_archive + (acq_elites, obj_evals)

        eval_budget -= 250

        print(acq_elites[1].T)

        gp_model = fit_gp_model(sol_archive, obj_archive)

    print("Exit Acquisition Loop")

    #### PREDICTION MAP
    archive = map_elites(archive, emitter, PRED_N_EVALS, predict_objective(gp_model), example_behavior_function, example_variation_function)

    print("[...] Terminate sail()")

    return archive

if __name__ == "__main__":
    sail()
###### Archive packages #####
import numpy as np
from ribs.emitters import GaussianEmitter
from ribs.archives import GridArchive


###### Import Custom Scripts ######
from acq_functions.acq_normal_distribution import acq_normal_distribution
from acq_functions.acq_ucb import acq_ucb
from example.example_functions import example_objective_function, example_behavior_function, example_variation_function
from utils.initialize_archive import initialize_archive
from utils.fit_gp_model import fit_gp_model
from utils.pprint import pprint
from utils.predict_objective import predict_objective
from map_elites import map_elites
from torch import float64

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

def sail():

    print("Initialize sail() [...]")

    obj_archive = GridArchive(
        solution_dim=SOL_DIMENSION,         # Dimension of solution vector
        dims=BHV_NUMBER_BINS,               # Discretization of behavioral bins
        ranges=BHV_VALUE_RANGE,             # Possible values for behavior vector
        qd_score_offset=-600)
    
    acq_archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600)

    obj_archive, init_solutions, init_obj_evals = initialize_archive(obj_archive, example_objective_function,example_behavior_function)

    init_bhv_evals = example_behavior_function(init_solutions)
    
    obj_archive.add(init_solutions, init_obj_evals.ravel(), init_bhv_evals)

    sol_array = []
    obj_array = []

    sol_array = init_solutions
    obj_array = init_obj_evals

    gp_model = fit_gp_model(sol_array, obj_array)

    acq_emitter = [
        GaussianEmitter(
        archive=acq_archive,
        sigma=0.5,
        bounds= SOL_VALUE_RANGE,
        batch_size=PARALLEL_BATCH_SIZE,
        initial_solutions=init_solutions,
    )]

    print(" ## Exit Initialization ##")
    print(" ## Enter Acquisition Loop ##")

    eval_budget = ACQ_N_OBJ_EVALS

    while(eval_budget-PARALLEL_BATCH_SIZE >= 0): # future addition: add threshhold condition for predictive performance of the model
        
        print("Enter ACQ Loop")
        print("Remaining Evals: " + str(eval_budget))

        acq_archive = map_elites(acq_archive, acq_emitter, gp_model, ACQ_N_MAP_EVALS, acq_ucb, example_behavior_function, example_variation_function)
        
        acq_elites = acq_archive.sample_elites(PARALLEL_BATCH_SIZE)     # Select acquisition elites (sobol sample in original paper)
        obj_evals = example_objective_function(acq_elites[0])
        
        obj_archive.add(acq_elites[0], obj_evals.ravel(), acq_elites[2])

        eval_budget -= 250 # PARALLEL_BATCH_SIZE

        sol_array = np.vstack((sol_array, acq_elites[0]), dtype=float)
        obj_array = np.vstack((obj_array, obj_evals.reshape(-1,1)), dtype=float)

        gp_model = fit_gp_model(sol_array, obj_array)

        print("Exit ACQ Loop")

    print(" ## Exit Acquisition Loop ##")
    print(" ## Enter Prediction Loop ##")

    obj_emitter = [
        GaussianEmitter(
        archive=obj_archive,
        sigma=0.5,
        bounds=SOL_VALUE_RANGE,
        batch_size=PARALLEL_BATCH_SIZE,
        initial_solutions=sol_array,
    )]

    obj_archive = map_elites(obj_archive, obj_emitter, gp_model, PRED_N_EVALS, predict_objective, example_behavior_function, example_variation_function)

    print("[...] Terminate sail()")

    return obj_archive

if __name__ == "__main__":
    sail()
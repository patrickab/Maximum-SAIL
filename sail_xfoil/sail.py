###### Archive packages #####
import numpy as np
from ribs.emitters import GaussianEmitter
from ribs.archives import GridArchive


###### Import Custom Scripts ######
from xfoil.simulate_airfoils import simulate_obj, simulate_bhv
from acq_functions.acq_ucb import acq_ucb
from gp.initialize_archive import initialize_archive
from gp.fit_gp_model import fit_gp_model
from gp.predict_objective import predict_objective
from map_elites import map_elites
from torch import float64

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
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

    np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=120)

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

    obj_archive, init_solutions, init_obj_evals, init_bhv_evals = initialize_archive(obj_archive, simulate_obj, simulate_bhv)

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
        bounds= np.array(SOL_VALUE_RANGE),
        batch_size=PARALLEL_BATCH_SIZE,
        initial_solutions=init_solutions,
    )]

    print("\n ## Exit Initialization ##")
    print(" ## Enter Acquisition Loop ##\n\n")

    eval_budget = ACQ_N_OBJ_EVALS

    while(eval_budget-PARALLEL_BATCH_SIZE > 0): # future addition: add threshhold condition for predictive performance of the model
        
        print("Remaining ACQ Evals: " + str(eval_budget))

        acq_archive = map_elites(acq_archive, acq_emitter, gp_model, ACQ_N_MAP_EVALS, acq_ucb, simulate_bhv)
        
        acq_elites = acq_archive.sample_elites(PARALLEL_BATCH_SIZE)     # Select acquisition elites (sobol sample in original paper)
        obj_evals = simulate_obj(acq_elites[0])
        
        obj_archive.add(acq_elites[0], obj_evals.ravel(), acq_elites[2])

        eval_budget -= PARALLEL_BATCH_SIZE

        sol_array = np.vstack((sol_array, acq_elites[0]), dtype=float64)
        obj_array = np.vstack((obj_array, obj_evals.reshape(-1,1)), dtype=float64)

        gp_model = fit_gp_model(sol_array, obj_array)

    print("\n\n ## Exit Acquisition Loop ##")
    print(" ## Enter Prediction Loop ##\n\n")

    obj_emitter = [
        GaussianEmitter(
        archive=obj_archive,
        sigma=0.5,
        bounds=SOL_VALUE_RANGE,
        batch_size=PARALLEL_BATCH_SIZE,
        initial_solutions=sol_array,
    )]

    obj_archive = map_elites(obj_archive, obj_emitter, gp_model, PRED_N_EVALS, predict_objective, simulate_bhv)

    print("[...] Terminate sail()")

    return obj_archive

if __name__ == "__main__":
    sail()
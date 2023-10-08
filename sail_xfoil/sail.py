###### Archive packages #####
import numpy as np
import subprocess
import time
import json
import gc
from ribs.emitters import GaussianEmitter
from ribs.archives import GridArchive

from memory_profiler import profile


###### Import Custom Scripts ######
from utils.benchmark_utils import sail_vanilla, sail_custom, sail_random, store_benchmark_data
from xfoil.simulate_airfoils import xfoil
from xfoil.generate_airfoils import generate_parsec_coordinates
from acq_functions.acq_ucb import acq_ucb
from gp.initialize_archive import initialize_archive
from gp.fit_gp_model import fit_gp_model
from gp.predict_objective import predict_objective
from map_elites import map_elites
from utils.pprint_nd import pprint, pprint_fstring
from numpy import float32, float16

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
ACQ_N_OBJ_EVALS = config.ACQ_N_OBJ_EVALS
ACQ_N_MAP_EVALS = config.ACQ_N_MAP_EVALS
PRED_N_EVALS = config.PRED_N_EVALS
BATCH_SIZE = config.BATCH_SIZE
SOL_DIMENSION = config.SOL_DIMENSION
BHV_ARCHIVE_DIMENSION = config.BHV_ARCHIVE_DIMENSION
BHV_VALUE_RANGE = config.BHV_VALUE_RANGE
BHV_NUMBER_BINS = config.BHV_NUMBER_BINS
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
TEST_RUNS = config.TEST_RUNS
BHV_DIMENSION = config.BHV_DIMENSION

import warnings
warnings.filterwarnings("ignore", message="CUDA initialization: The NVIDIA driver on your system is too old")
np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=120)

dtype = float16

def sail(initial_seed, sail_vanilla_flag=False, sail_custom_flag=False, sail_random_flag=False):

    print("Initialize sail() [...]")

    seed = initial_seed

    obj_archive = GridArchive(
        solution_dim=SOL_DIMENSION,         # Dimension of solution vector
        dims=BHV_NUMBER_BINS,               # Discretization of behavioral bins
        ranges=BHV_VALUE_RANGE,             # Possible values for behavior vector
        qd_score_offset=-600,
        threshold_min = -1,
        seed=seed
        )
    
    acq_archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = 0.2,
        seed=seed
        )
    

    pred_archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = -1,
        seed=seed,
        )
    
    obj_archive, init_solutions, init_obj_evals = initialize_archive(obj_archive, seed)

    gp_model = fit_gp_model(init_solutions, init_obj_evals)

    sol_array = np.array(init_solutions)
    obj_array = np.array(init_obj_evals)

    obj_elites = np.array([elite.solution for elite in obj_archive])
    obj_elites_acq = acq_ucb(obj_elites, gp_model)
    obj_elites_measures = np.array([elite.measures for elite in obj_archive])

    acq_archive.add(obj_elites, obj_elites_acq, obj_elites_measures)

    acq_emitter = [
        GaussianEmitter(
        archive=acq_archive,
        sigma=1,
        bounds= np.array(SOL_VALUE_RANGE),
        batch_size=BATCH_SIZE,
        initial_solutions=init_solutions, # these solutions are never used, as the archive is never empty - however, specification is required for initializing the GaussianEmitter class
        seed=seed
    )]

    print("\n ## Exit Initialization ##")
    print(" ## Enter Acquisition Loop ##\n\n")

    eval_budget = ACQ_N_OBJ_EVALS
    if sail_vanilla_flag:
        obj_archive, gp_model = sail_vanilla(acq_archive, obj_archive, gp_model, acq_emitter, sol_array, obj_array, eval_budget)
    if sail_custom_flag:
        obj_archive, gp_model = sail_custom(acq_archive, obj_archive, gp_model, acq_emitter, sol_array, obj_array, eval_budget)
    if sail_random_flag:
        obj_archive, gp_model = sail_random(acq_archive, obj_archive, gp_model, acq_emitter, sol_array, obj_array, eval_budget)

    print("\n\n ## Exit Acquisition Loop ##")
    print(" ## Enter Prediction Loop ##\n\n")


    pred_emitter = [
        GaussianEmitter(
        archive=obj_archive,
        sigma=0.5,
        bounds=SOL_VALUE_RANGE,
        batch_size=BATCH_SIZE,
        initial_solutions=[elite.solution for elite in obj_archive],
        seed=seed
    )]

    pred_archive, new_elites = map_elites(pred_archive, pred_emitter, gp_model, PRED_N_EVALS, predict_objective)

    print("[...] Terminate sail()")
    gc.collect()

    return obj_archive, pred_archive


if __name__ == "__main__":

    exec_start = time.time()

    mse_vanilla_array = np.empty(0, dtype=dtype)
    qd_vanilla_array = np.empty(0, dtype=dtype) # referring to verified_obj_archive
    percent_invalid_vanilla = np.empty(0, dtype=dtype) # predicted elites, that did not converge in xfoil

    mse_custom_array = np.empty(0, dtype=dtype)
    qd_custom_array = np.empty(0, dtype=dtype) # referring to verified_obj_archive
    percent_invalid_custom = np.empty(0, dtype=dtype) # predicted elites, that did not converge in xfoil

    mse_random_array = np.empty(0, dtype=dtype)
    qd_random_array = np.empty(0, dtype=dtype) # referring to verified_obj_archive
    percent_invalid_random = np.empty(0, dtype=dtype) # predicted elites, that did not converge in xfoil

    for i in range(TEST_RUNS):

        gc.collect()

        data = {}

        obj_archive, pred_archive = sail(i, sail_custom_flag=True)
        mse_custom, qd_custom, perc_invalid = store_benchmark_data(i, obj_archive, pred_archive, sail_vanilla_flag=True)
        mse_custom_array = np.append(mse_custom_array, mse_custom)
        qd_custom_array = np.append(qd_custom_array, qd_custom)
        percent_invalid_custom = np.append(percent_invalid_custom, perc_invalid)
        obj_archive.clear()
        pred_archive.clear()

        gc.collect()

        obj_archive, pred_archive = sail(i, sail_vanilla_flag=True)
        mse_vanilla, qd_vanilla, perc_invalid = store_benchmark_data(i, obj_archive, pred_archive, sail_vanilla_flag=True)
        mse_vanilla_array = np.append(mse_vanilla_array, mse_vanilla)
        qd_vanilla_array = np.append(qd_vanilla_array, qd_vanilla)
        percent_invalid_vanilla = np.append(percent_invalid_vanilla, perc_invalid)
        obj_archive.clear()
        pred_archive.clear()

        gc.collect()

        obj_archive, pred_archive = sail(i, sail_random_flag=True)
        mse_random, qd_random, perc_invalid = store_benchmark_data(i, obj_archive, pred_archive, sail_random_flag=True)
        mse_random_array = np.append(mse_random_array, mse_random)
        qd_random_array = np.append(qd_random_array, qd_random)
        percent_invalid_random = np.append(percent_invalid_random, perc_invalid)
        obj_archive.clear()
        pred_archive.clear()

        gc.collect()

        pprint_fstring(mse_custom_array, mse_vanilla_array, mse_random_array)
        pprint_fstring(qd_custom_array, qd_vanilla_array, qd_random_array)
        pprint_fstring(percent_invalid_custom, percent_invalid_vanilla, percent_invalid_random)

        gc.collect()

    subprocess.run(f"rm *.log *.dat", shell=True)
    subprocess.run(f"mv *.csv csv", shell=True)

    exec_end = time.time()
    exec_time = exec_end - exec_start
    print("\nExecution time (minutes): " + str(exec_time/60))
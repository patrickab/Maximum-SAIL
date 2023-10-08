###### Archive packages #####
import numpy as np
import subprocess
import time
import gc
from ribs.emitters import GaussianEmitter

###### Import Custom Scripts ######
from utils.while_loops import sail_vanilla, sail_custom, sail_random
from utils.benchmark_utils import benchmark_sail
from utils.utils import define_archives, define_emitter
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
SIGMA_EMITTER = config.SIGMA_EMITTER

import warnings
warnings.filterwarnings("ignore", message="CUDA initialization: The NVIDIA driver on your system is too old")
np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=120)

dtype = float16

def sail(initial_seed, sail_vanilla_flag=False, sail_custom_flag=False, sail_random_flag=False):

    print("Initialize sail() [...]")

    seed = initial_seed+10

    obj_archive, acq_archive, pred_archive = define_archives(seed)
    obj_archive, init_solutions, init_obj_evals = initialize_archive(obj_archive, seed)

    gp_model = fit_gp_model(init_solutions, init_obj_evals)

    sol_array = np.array(init_solutions)
    obj_array = np.array(init_obj_evals)


    print("\n ## Exit Initialization ##")
    print(" ## Enter Acquisition Loop ##\n\n")
    if sail_vanilla_flag:
        obj_archive, gp_model = sail_vanilla(acq_archive, obj_archive, gp_model, sol_array, obj_array)
        gc.collect()
    if sail_custom_flag:
        obj_archive, gp_model = sail_custom(acq_archive, obj_archive, gp_model, sol_array, obj_array)
        gc.collect()
    if sail_random_flag:
        obj_archive, gp_model = sail_random(acq_archive, obj_archive, gp_model, sol_array, obj_array)
        gc.collect()


    print("\n\n ## Exit Acquisition Loop ##")
    print(" ## Enter Prediction Loop ##\n\n")
    pred_emitter = define_emitter(init_solutions=[elite.solution for elite in obj_archive], pred_archive=pred_archive, seed=seed)
    pred_archive, new_elites = map_elites(pred_archive, pred_emitter, gp_model, PRED_N_EVALS, predict_objective)


    print("[...] Terminate sail()")
    gc.collect()

    return obj_archive, pred_archive


if __name__ == "__main__":

    exec_start = time.time()

    mse_custom_array, qd_custom_array, percent_invalid_custom = np.empty(0, dtype=dtype), np.empty(0, dtype=dtype), np.empty(0, dtype=dtype)
    mse_vanilla_array, qd_vanilla_array, percent_invalid_vanilla = np.empty(0, dtype=dtype), np.empty(0, dtype=dtype), np.empty(0, dtype=dtype)
    mse_random_array, qd_random_array, percent_invalid_random = np.empty(0, dtype=dtype), np.empty(0, dtype=dtype), np.empty(0, dtype=dtype)

    for i in range(TEST_RUNS):

        gc.collect()
        
        mse_custom_array, qd_custom_array, percent_invalid_custom = benchmark_sail(i, sail_custom_flag=True, mse_array=mse_custom_array, qd_array=qd_custom_array, percent_invalid_array=percent_invalid_custom)
        mse_vanilla_array, qd_vanilla_array, percent_invalid_vanilla = benchmark_sail(i, sail_vanilla_flag=True, mse_array=mse_vanilla_array, qd_array=qd_vanilla_array, percent_invalid_array=percent_invalid_vanilla)
        mse_random_array, qd_random_array, percent_invalid_random = benchmark_sail(i, sail_random_flag=True, mse_array=mse_random_array, qd_array=qd_random_array, percent_invalid_array=percent_invalid_random)

        pprint_fstring(mse_custom_array, mse_vanilla_array, mse_random_array)
        pprint_fstring(qd_custom_array, qd_vanilla_array, qd_random_array)
        pprint_fstring(percent_invalid_custom, percent_invalid_vanilla, percent_invalid_random)

        gc.collect()

    subprocess.run(f"rm *.log *.dat", shell=True)
    subprocess.run(f"mv *.csv csv", shell=True)

    exec_end = time.time()
    exec_time = exec_end - exec_start
    print("\nExecution time (minutes): " + str(exec_time/60))
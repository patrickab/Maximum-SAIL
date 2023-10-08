###### Archive packages #####
import numpy as np
import subprocess
import time
import gc

###### Import Custom Scripts ######
from utils.while_loops import sail_vanilla, sail_custom, sail_random
from utils.benchmark_utils import store_benchmark_data
from utils.utils import define_archives, define_emitter
from gp.initialize_archive import initialize_archive
from gp.fit_gp_model import fit_gp_model
from gp.predict_objective import predict_objective
from map_elites import map_elites
from utils.pprint_nd import pprint, pprint_fstring
from numpy import float16, float32

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
PRED_N_EVALS = config.PRED_N_EVALS
TEST_RUNS = config.TEST_RUNS

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
        obj_archive, gp_model = sail_vanilla(acq_archive=acq_archive, obj_archive=obj_archive, gp_model=gp_model, sol_array=sol_array, obj_array=obj_array)
        gc.collect()
    if sail_custom_flag:
        obj_archive, gp_model = sail_custom(acq_archive=acq_archive, obj_archive=obj_archive, gp_model=gp_model, sol_array=sol_array, obj_array=obj_array)
        gc.collect()
    if sail_random_flag:
        obj_archive, gp_model = sail_random(acq_archive=acq_archive, obj_archive=obj_archive, gp_model=gp_model, sol_array=sol_array, obj_array=obj_array)
        gc.collect()


    print("\n\n ## Exit Acquisition Loop ##")
    print(" ## Enter Prediction Loop ##\n\n")
    pred_emitter = define_emitter(init_solutions=[elite.solution for elite in obj_archive], archive=pred_archive, seed=seed)
    pred_archive, _ = map_elites(pred_archive, pred_emitter, gp_model, PRED_N_EVALS, predict_objective)
    _.clear()
    gc.collect()

    print("[...] Terminate sail()")
    gc.collect()

    return obj_archive, pred_archive


def benchmark_sail(i, mse_array, qd_array, percent_invalid_array, sail_custom_flag=False, sail_vanilla_flag=False, sail_random_flag=False):

    obj_archive, pred_archive = sail(i, sail_custom_flag=sail_custom_flag, sail_vanilla_flag=sail_vanilla_flag, sail_random_flag=sail_random_flag)

    mse, qd, perc_invalid = store_benchmark_data(i, obj_archive, pred_archive, sail_custom_flag=sail_custom_flag, sail_vanilla_flag=sail_vanilla_flag, sail_random_flag=sail_random_flag)

    mse_array = np.append(mse_array, mse)
    qd_array = np.append(qd_array, qd)
    percent_invalid_array = np.append(percent_invalid_array, perc_invalid)

    obj_archive.clear()
    pred_archive.clear()

    gc.collect()

    return mse_array, qd_array, percent_invalid_array


if __name__ == "__main__":

    exec_start = time.time()

    mse_custom_array, qd_custom_array, percent_invalid_custom = np.empty(0, dtype=dtype), np.empty(0, dtype=dtype), np.empty(0, dtype=dtype)
    mse_vanilla_array, qd_vanilla_array, percent_invalid_vanilla = np.empty(0, dtype=dtype), np.empty(0, dtype=dtype), np.empty(0, dtype=dtype)
    mse_random_array, qd_random_array, percent_invalid_random = np.empty(0, dtype=dtype), np.empty(0, dtype=dtype), np.empty(0, dtype=dtype)

    for i in range(TEST_RUNS):

        gc.collect()
        
        mse_custom_array, qd_custom_array, percent_invalid_custom = benchmark_sail(i, mse_array=mse_custom_array, qd_array=qd_custom_array, percent_invalid_array=percent_invalid_custom, sail_custom_flag=True)
        mse_vanilla_array, qd_vanilla_array, percent_invalid_vanilla = benchmark_sail(i, mse_array=mse_vanilla_array, qd_array=qd_vanilla_array, percent_invalid_array=percent_invalid_vanilla, sail_vanilla_flag=True)
        mse_random_array, qd_random_array, percent_invalid_random = benchmark_sail(i, mse_array=mse_random_array, qd_array=qd_random_array, percent_invalid_array=percent_invalid_random, sail_random_flag=True)

        pprint_fstring(mse_custom_array, mse_vanilla_array, mse_random_array)
        pprint_fstring(qd_custom_array, qd_vanilla_array, qd_random_array)
        pprint_fstring(percent_invalid_custom, percent_invalid_vanilla, percent_invalid_random)

        gc.collect()

    subprocess.run(f"rm *.log *.dat", shell=True)
    subprocess.run(f"mv *.csv csv", shell=True)

    exec_end = time.time()
    exec_time = exec_end - exec_start
    print("\nExecution time (minutes): " + str(exec_time/60))
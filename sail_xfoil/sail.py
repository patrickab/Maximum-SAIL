###### Import Foreign Packages #####
import numpy as np
import subprocess
import time
import gc

###### Import Custom Scripts ######
from utils.sail_loops import sail_vanilla, sail_custom, sail_random
from utils.benchmark_utils import store_benchmark_data
from utils.utils import define_archives, init_pred_archive
from gp.initialize_archive import initialize_archive
from gp.fit_gp_model import fit_gp_model
from gp.predict_objective import predict_objective
from utils.prediction_verification import prediction_verification_loop
from map_elites import map_elites
from utils.pprint_nd import pprint, pprint_fstring
from numpy import float16, float32

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
SIGMA_PRED_EMITTER = config.SIGMA_PRED_EMITTER
PRED_ELITE_REEVALS = config.PRED_ELITE_REEVALS
PRED_N_EVALS = config.PRED_N_EVALS
BATCH_SIZE = config.BATCH_SIZE
PRED_N_EVALS = config.PRED_N_EVALS
TEST_RUNS = config.TEST_RUNS

import warnings
warnings.filterwarnings("ignore", message="CUDA initialization: The NVIDIA driver on your system is too old")
np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=120)

pred_n_evals = PRED_N_EVALS//PRED_ELITE_REEVALS
if pred_n_evals % BATCH_SIZE != 0:
    ValueError("PRED_N_EVALS must be divisible by PRED_ELITE_REEVALS")

dtype = float16

def sail(initial_seed, sail_vanilla_flag=False, sail_custom_flag=False, sail_random_flag=False, pred_verific_flag=False, extra_evals=0):
    """
    Note: Extra Evals are only used if pred_verific_flag is set to True resulting in more than ACQ_N_OBJ_EVALS. In this case the extra evaluations are counted, returned & also given to subsequent sail runs
    """

    print("Initialize sail() [...]")
    seed = initial_seed+10

    obj_archive, acq_archive, pred_archive = define_archives(initial_seed=seed)
    obj_archive, init_solutions, init_obj_evals = initialize_archive(archive=obj_archive, seed=seed)

    gp_model = fit_gp_model(init_solutions, init_obj_evals)

    sol_array = np.array(init_solutions)
    obj_array = np.array(init_obj_evals)


    print("\n ## Exit Initialization ##")
    print(" ## Enter Acquisition Loop ##\n\n")
    if sail_vanilla_flag:
        obj_archive, gp_model = sail_vanilla(acq_archive=acq_archive, obj_archive=obj_archive, gp_model=gp_model, sol_array=sol_array, obj_array=obj_array, extra_evals=extra_evals)
        gc.collect()
    if sail_custom_flag:
        obj_archive, gp_model = sail_custom(acq_archive=acq_archive, obj_archive=obj_archive, gp_model=gp_model, sol_array=sol_array, obj_array=obj_array, extra_evals=extra_evals)
        gc.collect()
    if sail_random_flag:
        obj_archive, gp_model = sail_random(acq_archive=acq_archive, obj_archive=obj_archive, gp_model=gp_model, sol_array=sol_array, obj_array=obj_array, extra_evals=extra_evals)
        gc.collect()


    print("\n\n ## Exit Acquisition Loop ##")
    print(" ## Enter Prediction Loop ##\n\n")
    seed = acq_archive._seed
    pred_archive, pred_emitter = init_pred_archive(pred_archive, obj_archive, seed)

    if pred_verific_flag:
        pred_archive, extra_evals = prediction_verification_loop(pred_archive, obj_archive, pred_emitter, gp_model, sol_array, obj_array, extra_evals=extra_evals)
    else:
        pred_archive, new_elite_archive = map_elites(pred_archive, pred_emitter, gp_model, PRED_N_EVALS, predict_objective)
    gc.collect()


    print("[...] Terminate sail()")
    if pred_verific_flag:
        print(f"\n\n\nExtra evaluations: {extra_evals}\n\n\n")
        return obj_archive, pred_archive, extra_evals
    return obj_archive, pred_archive


def benchmark_sail(i, mse_array, qd_array, percent_invalid_array, sail_custom_flag=False, sail_vanilla_flag=False, sail_random_flag=False, pred_verific_flag=False, extra_evals=0):
    """
    WARNING: it is crucial to run pred_verific_flag first
    If pred_verific_flag additional evaluations are counted, returned & also given to subsequent sail runs
    """

    if pred_verific_flag:
        obj_archive, pred_archive, extra_evals = sail(i, sail_custom_flag=sail_custom_flag, sail_vanilla_flag=sail_vanilla_flag, sail_random_flag=sail_random_flag, pred_verific_flag=pred_verific_flag, extra_evals=extra_evals)
        print(f"Extra evaluations: {extra_evals}")
    else:
        obj_archive, pred_archive = sail(i, sail_custom_flag=sail_custom_flag, sail_vanilla_flag=sail_vanilla_flag, sail_random_flag=sail_random_flag, pred_verific_flag=pred_verific_flag, extra_evals=extra_evals)

    mse, qd, perc_invalid = store_benchmark_data(i, obj_archive, pred_archive, sail_custom_flag=sail_custom_flag, sail_vanilla_flag=sail_vanilla_flag, sail_random_flag=sail_random_flag, pred_verific_flag=pred_verific_flag)    

    mse_array = np.append(mse_array, mse)
    qd_array = np.append(qd_array, qd)
    percent_invalid_array = np.append(percent_invalid_array, perc_invalid)

    obj_archive.clear()
    pred_archive.clear()
    gc.collect()

    return mse_array, qd_array, percent_invalid_array, extra_evals


if __name__ == "__main__":

    exec_start = time.time()
    mse_random_array, qd_random_array, percent_random_invalid = np.empty(0, dtype=dtype), np.empty(0, dtype=dtype), np.empty(0, dtype=dtype)
    mse_custom_array, qd_custom_array, percent_custom_invalid = np.empty(0, dtype=dtype), np.empty(0, dtype=dtype), np.empty(0, dtype=dtype)
    mse_custom_reeval_array, qd_custom_reeval_array, percent_custom_reeval_invalid = np.empty(0, dtype=dtype), np.empty(0, dtype=dtype), np.empty(0, dtype=dtype)
    extra_evals_array = np.empty(0, dtype=dtype)

    for i in range(TEST_RUNS):

        gc.collect()
        
        mse_custom_reeval_array, qd_custom_reeval_array, percent_custom_reeval_invalid, extra_evals = benchmark_sail(i, mse_array=mse_custom_reeval_array, qd_array=qd_custom_reeval_array, percent_invalid_array=percent_custom_reeval_invalid, sail_custom_flag=True, pred_verific_flag=True)
        pprint_fstring(mse_custom_reeval_array, qd_custom_reeval_array, percent_custom_reeval_invalid)
        mse_custom_array, qd_custom_array, percent_custom_invalid, extra_evals = benchmark_sail(i, mse_array=mse_custom_array, qd_array=qd_custom_array, percent_invalid_array=percent_custom_invalid, sail_custom_flag=True, extra_evals=extra_evals)
        mse_random_array, qd_random_array, percent_random_invalid, extra_evals = benchmark_sail(i, mse_array=mse_random_array, qd_array=qd_random_array, percent_invalid_array=percent_random_invalid, sail_random_flag=True, extra_evals=extra_evals)

        pprint_fstring(mse_random_array,mse_custom_array, mse_custom_reeval_array)
        pprint_fstring(qd_random_array,qd_custom_array, qd_custom_reeval_array)
        pprint_fstring(percent_random_invalid, percent_custom_invalid, percent_custom_reeval_invalid)
        print("Extra evaluations: " + str(extra_evals_array))

        extra_evals_array = np.append(extra_evals_array, extra_evals)
        extra_evals = 0 # not sure if necessary, but better safe then sorry
        gc.collect()

    subprocess.run(f"rm *.log *.dat", shell=True)
    subprocess.run(f"mv *.csv csv", shell=True)
    exec_end = time.time()
    exec_time = exec_end - exec_start
    print("\nExecution time (minutes): " + str(exec_time/60))
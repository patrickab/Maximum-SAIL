###### Import Foreign Packages #####
from ribs.emitters import GaussianEmitter
import numpy as np
import subprocess
import datetime
import time
import gc
import os

###### Import Custom Scripts ######
from sail_runner import SailRun, run_custom_sail, run_vanilla_sail, run_random_sail, prediction_verification_loop #run_vanilla_sail, run_random_sail, 
from gp.predict_objective import predict_objective
from utils.pprint_nd import pprint
from map_elites import map_elites


###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
MAX_PRED_VERIFICATION = config.MAX_PRED_VERIFICATION
SIGMA_PRED_EMITTER = config.SIGMA_PRED_EMITTER
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SIGMA_EMITTER = config.SIGMA_EMITTER
PRED_N_EVALS = config.PRED_N_EVALS
BATCH_SIZE = config.BATCH_SIZE
TEST_RUNS = config.TEST_RUNS


import warnings
warnings.filterwarnings("ignore", message="CUDA initialization: The NVIDIA driver on your system is too old")
np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=120)

# global variable used for building dynamic folder structure for benchmarks
benchmark_domains = []

def sail(initial_seed, sail_vanilla_flag=False, sail_custom_flag=False, sail_random_flag=False, pred_verific_flag=False, greedy_flag=False, explore_flag=False, extra_evals=None):

    """
    Note: Extra Evals are only used if pred_verific_flag is set to True resulting in more than ACQ_N_OBJ_EVALS. In this case the extra evaluations are counted, returned & also given to subsequent sail runs
    """

    current_run = SailRun(initial_seed, sail_vanilla_flag=sail_vanilla_flag, sail_custom_flag=sail_custom_flag, sail_random_flag=sail_random_flag, pred_verific_flag=pred_verific_flag, greedy_flag=greedy_flag, explore_flag=explore_flag) 

    print("\n ## Exit Initialization ##")
    print(" ## Enter Acquisition Loop ##\n\n")

    if sail_custom_flag:
        run_custom_sail(current_run)
        gc.collect()

    if sail_vanilla_flag:
        run_vanilla_sail(current_run)
        gc.collect()

    if sail_random_flag:
        run_random_sail(current_run)
        gc.collect()

    global benchmark_domains
    benchmark_domains.append(current_run.domain)


    print("\n\n ## Exit Acquisition Loop ##")
    print(" ## Enter Prediction Loop ##\n\n")

    if current_run.pred_verific_flag:
        pred_archive, extra_evals = prediction_verification_loop(current_run)
    else:
        pred_archive, new_elite_archive = map_elites(current_run, target_archive=pred_archive, n_evals=PRED_N_EVALS, fuct_obj=predict_objective, pred_flag=True)

    gc.collect()
    print("[...] Terminate sail()")

    return current_run.extra_evals



if __name__ == "__main__":

    exec_start = time.time()

    for i in range(TEST_RUNS):

        gc.collect()
        
        sail(initial_seed=i, sail_custom_flag=True, pred_verific_flag=True, greedy_flag=True)
        sail(initial_seed=i, sail_custom_flag=True, pred_verific_flag=False, greedy_flag=True)
        sail(initial_seed=i, sail_custom_flag=True, pred_verific_flag=True, explore_flag=True)
        sail(initial_seed=i, sail_custom_flag=True, pred_verific_flag=False, explore_flag=True)
        # sail(initial_seed=i, sail_vanilla_flag=True)
        # sail(initial_seed=i, sail_random_flag=True)
        # #extra_evals_2 = sail(initial_seed=i, sail_vanilla_flag=True, pred_verific_flag=True, explore_flag=True)
        # extra_evals = max(extra_evals_1, extra_evals_2)
        # sail(initial_seed=i, sail_vanilla_flag=True, extra_evals=extra_evals)
        # sail(initial_seed=i, sail_custom_flag=True, extra_evals=extra_evals)
        # sail(initial_seed=i, sail_random_flag=True, extra_evals=extra_evals)
        # extra_evals = 0 # not sure if necessary, but better safe then sorry
        gc.collect()

    benchmark_filepaths = " ".join(["imgs/" + benchmark_domain for benchmark_domain in benchmark_domains])
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M")
    os.makedirs(timestamp)
    os.makedirs("json") 
    subprocess.run("mv *.json json", shell=True)
    os.makedirs("csv")
    subprocess.run("mv *.csv csv", shell=True)
    subprocess.run(f"cp config/config.ini {timestamp}/reproduction_info.txt", shell=True)
    subprocess.run(f'mv json csv {benchmark_filepaths} {timestamp}', shell=True)
    if not os.path.exists("benchmarks"): os.makedirs("benchmarks")
    subprocess.run(f"mv {timestamp} benchmarks", shell=True)

    subprocess.run(f"rm *.log *.dat", shell=True)

    exec_end = time.time()
    exec_time = exec_end - exec_start
    print("\nExecution time (minutes): " + str(exec_time/60))
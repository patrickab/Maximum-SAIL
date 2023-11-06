import numpy as np
import subprocess
import datetime
import time
import PIL
import gc
import os

###### Import Custom Scripts ######
from sail_runs import run_custom_sail, run_vanilla_sail, run_random_sail
from sail_runner import SailRun, evaluate_prediction_archive, store_final_data
from map_elites import map_elites

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
PREDICTION_VERIFICATIONS = config.PREDICTION_VERIFICATIONS
PRED_N_OBJ_EVALS = config.PRED_N_OBJ_EVALS
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SIGMA_EMITTER = config.SIGMA_EMITTER
BATCH_SIZE = config.BATCH_SIZE
TEST_RUNS = config.TEST_RUNS


import warnings
warnings.filterwarnings("ignore", message="CUDA initialization: The NVIDIA driver on your system is too old")
np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=120)

# global variable used for building dynamic folder structure for benchmarks
benchmark_domains = []

if (BATCH_SIZE%2)!=0 or ((PRED_N_OBJ_EVALS//PREDICTION_VERIFICATIONS)%2)!=0:
    raise ValueError("BATCH_SIZE and MAX_PRED_VERIFICATION//PEDICTION_VERIFICATION must be even numbers")

def sail(initial_seed, acq_ucb_flag=False, acq_mes_flag=False, sail_vanilla_flag=False, sail_custom_flag=False, sail_random_flag=False, pred_verific_flag=False, greedy_flag=False, hybrid_flag=False, random_init=False, mes_init=False):

    """
    Main function for running SAIL. This function is used for controlling behavior of SailRun objects.

    Parameters

        initial_seed : used for seeding - among each benchmark iteration, all algorithms are seeded with the same value. This value is incremented by the number of TEST_RUNS, to ensure a unique sequence of seeds for each benchmark iteration, identical across all benchmarked algorithms.
    
        acq_ucb_flag : boolean flag for running SAIL with UCB acquisition function
        acq_mes_flag : boolean flag for running SAIL with MES acquisition function
    
        greedy_flag: boolean flag for sampling only highest performing solutions from new elites archive (100% exploitation)
        hybrid_flag: boolean flag for sampling certain percentage of highest performing solutions, certain percentage of new bin solutions (xx.xx% exploitation, 100-xx.xx% exploration)
    
        random_init: boolean flag for initializing target archive with quasi-random sobol samples
        mes_init: boolean flag for initializing target archive with MES samples
    
        sail_vanilla_flag : boolean flag for running SAIL with vanilla behavior
            - before each MAP-Loop, initialize target archive with objective elites
        sail_custom_flag : boolean flag for running SAIL with custom behavior 
            - before each MAP-Loop, initialize target archive with objective elites & updated target elites
            - requires selection of: (greedy_flag or hybrid_flag) and (random_init or mes_init)
            - offers possibility of using pred_verific_flag
        sail_random_flag : boolean flag for running SAIL with uniform random solutions

    """

    current_run = SailRun(initial_seed, acq_ucb_flag=acq_ucb_flag, acq_mes_flag=acq_mes_flag, sail_vanilla_flag=sail_vanilla_flag, sail_custom_flag=sail_custom_flag, sail_random_flag=sail_random_flag, pred_verific_flag=pred_verific_flag, greedy_flag=greedy_flag, hybrid_flag=hybrid_flag, random_init=random_init, mes_init=mes_init) 

    print("\n ## Exit Initialization ##")
    print(" ## Enter Acquisition Loop ##\n\n")

    if sail_vanilla_flag:
        run_vanilla_sail(current_run)
        gc.collect()

    if sail_custom_flag:
        run_custom_sail(current_run, acq_loop=True)
        gc.collect()

    if sail_random_flag:
        run_random_sail(current_run)
        gc.collect()

    global benchmark_domains
    benchmark_domains.append(current_run.domain)


    print("\n\n ## Exit Acquisition Loop ##")
    print(" ## Enter Prediction Loop ##\n\n")

    acq_elites_df = current_run.acq_archive.as_pandas(include_solutions=True)
    acq_elites_solutions = acq_elites_df.solution_batch()
    acq_elites_measures = acq_elites_df.measures_batch()

    obj_elites_df = current_run.obj_archive.as_pandas(include_solutions=True)
    obj_elites_solutions = obj_elites_df.solution_batch()
    obj_elites_measures = obj_elites_df.measures_batch()

    current_run.update_archive(candidate_sol=acq_elites_solutions, candidate_bhv=acq_elites_measures, pred_flag=True)
    current_run.update_archive(candidate_sol=obj_elites_solutions, candidate_bhv=obj_elites_measures, pred_flag=True)

    if current_run.pred_verific_flag:
        run_custom_sail(current_run, pred_loop=True)
    else:
        map_elites(current_run, pred_flag=True)

    evaluate_prediction_archive(current_run)
    store_final_data(current_run)
    gc.collect()

    print("[...] Terminate sail()")

    return


if __name__ == "__main__":

    subprocess.run("clean", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # custom bash command to remove all files from last run
    exec_start = time.time()

    for i in range(TEST_RUNS):

        gc.collect()

        benchmark_domains = []
        
        sail(initial_seed=i, sail_custom_flag=True, pred_verific_flag=True, greedy_flag=True, acq_mes_flag=True)
        sail(initial_seed=i, sail_custom_flag=True, pred_verific_flag=True, hybrid_flag=True, acq_ucb_flag=True, mes_init=True)
        #sail(initial_seed=i, sail_custom_flag=True, pred_verific_flag=True, hybrid_flag=True, acq_ucb_flag=True, random_init=True)
        #sail(initial_seed=i, sail_vanilla_flag=True, acq_ucb_flag=True)
        gc.collect()

        img_filenames = [f"imgs/final_heatmaps_{i}_{benchmark_domain}.png" for benchmark_domain in benchmark_domains]
        imgs = [PIL.Image.open(img) for img in img_filenames]
        imgs_comb = np.vstack([img for img in imgs])
        imgs_comb = PIL.Image.fromarray(imgs_comb)
        imgs_comb.save(f'imgs/final_heatmaps_{i}.png')


    benchmark_filepaths = " ".join(["imgs/" + benchmark_domain for benchmark_domain in benchmark_domains])
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d%H%M")
    os.makedirs(timestamp)
    subprocess.run(f"cp config/config.ini {timestamp}/reproduction_info.txt", shell=True)
    subprocess.run(f'mv *.csv csv', shell=True)
    subprocess.run(f'mv csv *.mp4 imgs stats_log {timestamp}', shell=True)
    if not os.path.exists("benchmarks"): os.makedirs("benchmarks")
    subprocess.run(f"mv {timestamp} benchmarks", shell=True)

    subprocess.run(f"rm *.log *.dat", shell=True)

    exec_end = time.time()
    exec_time = exec_end - exec_start
    print("\nExecution time (minutes): " + str(exec_time/60))
"""
Main function for running SAIL. This function is used for controlling behavior of SailRun objects.

Parameters

    initial_seed : used for seeding

    acq_ucb_flag : boolean flag for running SAIL with UCB acquisition function
    acq_mes_flag : boolean flag for running SAIL with MES acquisition function

    mes_init    : boolean flag for performing MES initialization before UCB loop

    sail_vanilla_flag : boolean flag for running SAIL with vanilla behavior
    sail_custom_flag  : boolean flag for running SAIL with custom behavior 
    sail_random_flag  : boolean flag for running SAIL with uniform random solutions

    pred_verific_flag : boolean flag for using prediction verification loop before returning prediction archive
"""

import numpy as np
import subprocess
import datetime
import time
import PIL
import gc
import os

###### Import Custom Scripts ######
from sail_variants import run_custom_sail, run_vanilla_sail, run_random_search, run_botorch_acqf
from sail_run import SailRun, evaluate_prediction_archive, store_final_data
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

np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=120)

def sail(initial_seed, acq_ucb_flag=False, acq_mes_flag=False, sail_vanilla_flag=False, sail_custom_flag=False, botorch_flag=False, sail_random_flag=False, pred_verific_flag=False, mes_init=False):

    current_run = SailRun(initial_seed, acq_ucb_flag=acq_ucb_flag, acq_mes_flag=acq_mes_flag, sail_vanilla_flag=sail_vanilla_flag, sail_custom_flag=sail_custom_flag, botorch_flag=botorch_flag, sail_random_flag=sail_random_flag, pred_verific_flag=pred_verific_flag, mes_init=mes_init) 

    if sail_vanilla_flag:
        run_vanilla_sail(current_run)
        gc.collect()

    if sail_custom_flag:
        run_custom_sail(current_run, acq_loop=True)
        gc.collect()

    if botorch_flag:
        run_botorch_acqf(current_run)


    if sail_random_flag:
        run_random_search(current_run)
        gc.collect()

    benchmark_domain = current_run.domain

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

    return benchmark_domain


def main():

    exec_start = time.time()

    for i in range(TEST_RUNS):
        gc.collect()

        start_time = time.time()

        benchmark_domain = sail(initial_seed=i, botorch_flag=True, acq_mes_flag=True)

        end_time = time.time()
        duration = end_time - start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        duration_str = f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}"

        with open("stats_log", "a") as file: 
            file.write(duration_str)

        gc.collect()

        img_filenames = [f"imgs/final_heatmaps_{i}_{benchmark_domain}.png"]
        imgs = [PIL.Image.open(img) for img in img_filenames]
        imgs_comb = np.vstack([img for img in imgs])
        imgs_comb = PIL.Image.fromarray(imgs_comb)
        imgs_comb.save(f'imgs/final_heatmaps_{i}.png')

    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y%m%d")
    os.makedirs(timestamp)
    subprocess.run(f"cp config/config.ini {timestamp}/reproduction_info.txt", shell=True)
    subprocess.run(f'mv *.csv csv', shell=True)
    subprocess.run(f'mv csv *.mp4 imgs stats_log mes-vs-botorch* {timestamp}', shell=True)
    if not os.path.exists("benchmarks"): os.makedirs("benchmarks")
    subprocess.run(f"mv {timestamp} benchmarks", shell=True)

    subprocess.run(f"rm *.log *.dat", shell=True)

    exec_end = time.time()
    exec_time = exec_end - exec_start
    print("\nExecution time (minutes): " + str(exec_time/60))


if __name__ == "__main__":
    main()


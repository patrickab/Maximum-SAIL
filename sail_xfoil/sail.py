###### Import Foreign Packages #####
from ribs.emitters import GaussianEmitter
from ribs.archives import GridArchive
import numpy as np
import subprocess
import datetime
import time
import gc
import os

###### Import Custom Scripts ######
from sail_runner import SailRun, run_custom_sail, run_vanilla_sail, run_random_sail, maximize_obj_improvement, eval_max_obj_improvement
from gp.predict_objective import predict_objective
from map_elites import map_elites
from utils.pprint_nd import pprint, pprint_fstring


###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
MAX_PRED_VERIFICATION = config.MAX_PRED_VERIFICATION
SIGMA_PRED_EMITTER = config.SIGMA_PRED_EMITTER
PRED_ELITE_REEVALS = config.PRED_ELITE_REEVALS
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SIGMA_EMITTER = config.SIGMA_EMITTER
PRED_N_EVALS = config.PRED_N_EVALS
BATCH_SIZE = config.BATCH_SIZE
TEST_RUNS = config.TEST_RUNS


import warnings
warnings.filterwarnings("ignore", message="CUDA initialization: The NVIDIA driver on your system is too old")
np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=120)

def sail(initial_seed, sail_vanilla_flag=False, sail_custom_flag=False, sail_random_flag=False, pred_verific_flag=False, extra_evals=None):
    """
    Note: Extra Evals are only used if pred_verific_flag is set to True resulting in more than ACQ_N_OBJ_EVALS. In this case the extra evaluations are counted, returned & also given to subsequent sail runs
    """

    current_run = SailRun(initial_seed, sail_vanilla_flag=sail_vanilla_flag, sail_custom_flag=sail_custom_flag, sail_random_flag=sail_random_flag, pred_verific_flag=pred_verific_flag) 

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

    print("\n\n ## Exit Acquisition Loop ##")
    print(" ## Enter Prediction Loop ##\n\n")

    pred_archive, pred_emitter = init_pred_archive(current_run.pred_archive, current_run.obj_archive, current_run.current_seed)

    if current_run.pred_verific_flag:
        pred_archive, extra_evals = prediction_verification_loop(current_run, pred_archive, pred_emitter)
    else:
        pred_archive, new_elite_archive = map_elites(current_run, target_archive=pred_archive, emitter=pred_emitter, 
                                            n_evals=PRED_N_EVALS, fuct_obj=predict_objective, pred_flag=True)

    gc.collect()
    print("[...] Terminate sail()")

    return current_run.extra_evals


def prediction_verification_loop(self: SailRun, pred_archive: GridArchive, pred_emitter: GaussianEmitter):

    print("\n\n ## Enter Prediction Verification Loop##")
    extra_evals = self.extra_evals
    pred_n_evals = PRED_N_EVALS//(PRED_ELITE_REEVALS) # +1 because after the loop predictions with map_elites is called once more
    obj_n_evals = MAX_PRED_VERIFICATION//PRED_ELITE_REEVALS
    dummy_elites = [elite.solution for elite in self.obj_archive][:BATCH_SIZE]
    total_pred_evals = PRED_N_EVALS
    flag = True

    while total_pred_evals > pred_n_evals:

        total_pred_evals -= pred_n_evals
        emitter = update_emitter(self, pred_archive, dummy_elites)
        old_elites = np.array([(elite.solution, elite.index, elite.objective, elite.measures) for elite in self.obj_archive], 
                        dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])

        if flag: new_elite_archive = None 
        else: new_elite_archive 
        flag=False

        pred_archive, new_elite_archive = map_elites(self, target_archive=pred_archive, emitter=emitter, 
                                                     n_evals=pred_n_evals, fuct_obj=predict_objective, 
                                                     new_elite_archive=new_elite_archive, pred_flag=True)
            
        # maximize_obj_improvement returns elites sorted in descending order (acq is obj within acq loop)
        improved_elites, new_elites, n_improvements = maximize_obj_improvement(new_elite_archive, old_elites) 

        # evaluate improved_elites & new_elites
        # update obj_archive and gp_model inside eval_max_obj_improvement()
        new_elite_sol, new_elite_obj, new_elite_bhv, pred_archive, gp_model = eval_max_obj_improvement(self, improved_elites, new_elites, old_elites, n_obj_evals=obj_n_evals, 
                                                                            emitter=emitter, target_archive=pred_archive, n_map_evals=pred_n_evals, 
                                                                            fuct_obj=predict_objective, new_elite_archive=new_elite_archive,
                                                                            explore_flag=True, greedy_flag=False)

    new_elite_archive.clear()
    gc.collect()

    # ToDo: ensure that exactly MAX_PRED_VERIFICATIONs are always evaluated
    if extra_evals < MAX_PRED_VERIFICATION:
        print("\n\n\nMaximum Pred Verifications not reached\n\n\n")
    if extra_evals > MAX_PRED_VERIFICATION:
        print("\n\n\nMaximum Pred Verifications exceeded\n\n\n")

    return pred_archive, extra_evals                                                                                                        # communicate extra evaluations


def update_emitter(self: SailRun, archive, dummy_elites):
    """
    Input: Updated Archive
    Output: Gaussian Emitter
    """
    emitter = [
        GaussianEmitter(
        archive=archive,
        sigma=SIGMA_PRED_EMITTER,
        bounds= np.array(SOL_VALUE_RANGE),
        batch_size=BATCH_SIZE,
        initial_solutions=dummy_elites, # these solutions are never used, as the archive is never empty - however, specification is required for initializing the GaussianEmitter class
        seed=self.update_seed()
    )]
    return emitter


def init_pred_archive(pred_archive, obj_archive, seed, sigma_emitter=SIGMA_PRED_EMITTER):
    """
    - Stores Obj Elites in Pred Archive
    - Generates Emitter for Pred Archive
    """
    pred_archive.add([elite.solution for elite in obj_archive], [elite.objective for elite in obj_archive], [elite.measures for elite in obj_archive])
    pred_emitter = generate_emitter(init_solutions=[elite.solution for elite in obj_archive], archive=pred_archive, seed=seed, sigma_emitter=sigma_emitter)
    return pred_archive, pred_emitter


def generate_emitter(init_solutions, archive, seed, sigma_emitter=SIGMA_EMITTER, sol_value_range=None):
    """Reduces Overhead"""

    if sol_value_range is None:
        sol_value_range = SOL_VALUE_RANGE

    emitter = [
        GaussianEmitter(
        archive=archive,
        sigma=sigma_emitter,
        bounds= np.array(sol_value_range),
        batch_size=BATCH_SIZE,
        initial_solutions=init_solutions,
        seed=seed
    )]

    return emitter


if __name__ == "__main__":

    exec_start = time.time()

    for i in range(TEST_RUNS):

        gc.collect()
        
        benchmark_domains = ["custom", "vanilla", "prediction_verification", "random"]
        extra_evals = sail(initial_seed=i, sail_custom_flag=True, pred_verific_flag=True)
        sail(initial_seed=i, sail_vanilla_flag=True, extra_evals=extra_evals)
        sail(initial_seed=i, sail_custom_flag=True, extra_evals=extra_evals)
        sail(initial_seed=i, sail_random_flag=True, extra_evals=extra_evals)
        extra_evals = 0 # not sure if necessary, but better safe then sorry
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
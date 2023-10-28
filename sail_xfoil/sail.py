###### Import Foreign Packages #####
import numpy as np
import subprocess
import datetime
import time
import PIL
import gc
import os

###### Import Custom Scripts ######
from sail_runner import SailRun, run_vanilla_sail #run_vanilla_sail, run_random_sail, 
from run_custom_sail import run_custom_sail, prediction_verification_loop
from utils.anytime_archive_visualizer import archive_visualizer
from gp.predict_objective import predict_objective
from utils.utils import eval_xfoil_loop
from utils.pprint_nd import pprint
from map_elites import map_elites


###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
PREDICTION_VERIFICATIONS = config.PREDICTION_VERIFICATIONS
MAX_PRED_VERIFICATION = config.MAX_PRED_VERIFICATION
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

if (BATCH_SIZE%2)!=0 or ((MAX_PRED_VERIFICATION//PREDICTION_VERIFICATIONS)%2)!=0:
    raise ValueError("BATCH_SIZE and MAX_PRED_VERIFICATION//PEDICTION_VERIFICATION must be even numbers")

def sail(initial_seed, sail_vanilla_flag=False, sail_custom_flag=False, sail_random_flag=False, pred_verific_flag=False, greedy_flag=False, explore_flag=False, hybrid_flag=False):

    """
    Note: Extra Evals are only used if pred_verific_flag is set to True resulting in more than ACQ_N_OBJ_EVALS. In this case the extra evaluations are counted, returned & also given to subsequent sail runs
    """

    current_run = SailRun(initial_seed, sail_vanilla_flag=sail_vanilla_flag, sail_custom_flag=sail_custom_flag, sail_random_flag=sail_random_flag, pred_verific_flag=pred_verific_flag, greedy_flag=greedy_flag, explore_flag=explore_flag, hybrid_flag=hybrid_flag) 

    print("\n ## Exit Initialization ##")
    print(" ## Enter Acquisition Loop ##\n\n")

    if sail_custom_flag:
        run_custom_sail(current_run)
        gc.collect()

    # if sail_vanilla_flag:
    #     run_vanilla_sail(current_run)
    #     gc.collect()

    # if sail_random_flag:
    #     run_random_sail(current_run)
    #     gc.collect()

    global benchmark_domains
    benchmark_domains.append(current_run.domain)


    print("\n\n ## Exit Acquisition Loop ##")
    print(" ## Enter Prediction Loop ##\n\n")

    if current_run.pred_verific_flag:
        prediction_verification_loop(current_run)
    else:
        map_elites(current_run, target_function=predict_objective, pred_flag=True)

    evaluate_predictions(current_run)
    store_final_data(current_run)
    gc.collect()

    print("[...] Terminate sail()")

    return


def store_final_data(self: SailRun):

    archive_visualizer(self=self, archive=self.obj_archive, prefix="obj", name="Objective Archive", min_val=1.0, max_val=5)
    archive_visualizer(self=self, archive=self.acq_archive, prefix="acq", name="Acquisition Archive", min_val=1.0, max_val=5)
    archive_visualizer(self=self, archive=self.pred_archive, prefix="pred", name="Prediction Archive (unevaluated)", min_val=1.0, max_val=5)
    archive_visualizer(self=self, archive=self.evaluated_predictions_archive, prefix="evaluted_pred", name="Prediction Archive (evaluated)", min_val=1.0, max_val=5)
    archive_visualizer(self=self, archive=self.prediction_error_archive, prefix="error", name="Prediction Error Archive (percentual)", min_val=0, max_val=0.1)

    initial_seed = self.initial_seed
    domain = self.domain

    img_filenames = [f"imgs/{domain}/{initial_seed}/final_{initial_seed}_{domain}_{prefix}_heatmap.png" for prefix in ["obj", "acq", "pred", "evaluted_pred", "error"]]
    imgs = [PIL.Image.open(img) for img in img_filenames]
    imgs_comb = np.hstack([img for img in imgs])
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    imgs_comb.save(f'imgs/final_heatmaps_{initial_seed}_{domain}.png')

    subprocess.run(f"rm imgs/{domain}/{initial_seed}/*.png", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    obj_dataframe = self.obj_archive.as_pandas(include_solutions=True)
    obj_dataframe.to_csv(f"{initial_seed}_{domain}_obj_archive.csv", index=False)

    acq_dataframe = self.acq_archive.as_pandas(include_solutions=True)
    acq_dataframe.to_csv(f"{initial_seed}_{domain}_acq_archive.csv", index=False)

    pred_dataframe = self.pred_archive.as_pandas(include_solutions=True)
    pred_dataframe.to_csv(f"{initial_seed}_{domain}_pred_archive.csv", index=False)
    
    # This archive contains only converged predictions & their true objective values
    evaluated_pred_dataframe = self.evaluated_predictions_archive.as_pandas(include_solutions=True)
    evaluated_pred_dataframe.to_csv(f"{initial_seed}_{domain}_evaluated_pred_archive.csv", index=False)

    error_dataframe = self.prediction_error_archive.as_pandas(include_solutions=True)
    error_dataframe.to_csv(f"{initial_seed}_{domain}_error_archive.csv", index=False)


def evaluate_predictions(self: SailRun):

    """
    Evaluate the predictions of the prediction archive.
    This is done to determine the quality of results.
    """

    print("Evaluate Prediction Archive")

    # Extract all elites from the prediction archive - (sorted by objective for nice visual effect during evaluation)
    unevaluated_prediction_elites = sorted(self.pred_archive, key=lambda x: x.objective, reverse=True)[:self.pred_archive.stats.num_elites]
    unevaluated_prediction_elites = np.array([(elite.solution, elite.index, elite.objective, elite.measures) for elite in unevaluated_prediction_elites], dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])
    unevaluated_prediction_solutions = unevaluated_prediction_elites['solution']
    unevaluated_prediction_measures = np.vstack(unevaluated_prediction_elites['behavior'])
    eval_xfoil_loop(self, solution_batch=unevaluated_prediction_solutions, measures_batch=unevaluated_prediction_measures, evaluate_prediction_archive=True, candidate_targetvalues=unevaluated_prediction_elites['objective'])

    # Extract all elites from the evaluated predictions archive - (sorted by index for comparison)
    evaluated_prediction_elites = sorted(self.evaluated_predictions_archive, key=lambda x: x.index)[:self.evaluated_predictions_archive.stats.num_elites]
    evaluated_prediction_elites = np.array([(elite.solution, elite.index, elite.objective, elite.measures) for elite in evaluated_prediction_elites], dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])
    unevaluated_prediction_elites = unevaluated_prediction_elites[np.argsort(unevaluated_prediction_elites['index'])]

    # Calculate mask for converged prediction elites
    is_converged_prediction_elite = np.isin(unevaluated_prediction_elites['index'], evaluated_prediction_elites['index'])

    # Extract converged prediction elites
    converged_unevaluated_prediction_elites = unevaluated_prediction_elites[is_converged_prediction_elite]
    converged_evaluated_prediction_elites = evaluated_prediction_elites

    prediction_error = converged_unevaluated_prediction_elites['objective'] - converged_evaluated_prediction_elites['objective']
    percentual_error = np.abs(prediction_error)/converged_evaluated_prediction_elites['objective']
    mean_percentual_error = np.mean(percentual_error)
    mae_error = np.mean(np.abs(prediction_error))
    mse_error = np.mean(np.square(prediction_error))

    self.evaluated_predictions_archive.add(np.vstack(converged_evaluated_prediction_elites['solution']), converged_evaluated_prediction_elites['objective'], np.vstack(converged_evaluated_prediction_elites['behavior']))
    self.prediction_error_archive.add(np.vstack(converged_unevaluated_prediction_elites['solution']), percentual_error, np.vstack(converged_unevaluated_prediction_elites['behavior']))

    percentual_errors_greater_than_005 = np.sum(np.abs(prediction_error)/converged_evaluated_prediction_elites['objective'] > 0.05)
    error_str = f"Initial Seed: {self.initial_seed}  Domain: {self.domain}  MAE Error: {mae_error}  MSE Error: {mse_error} \nPrediction Errors: \n{np.array2string(prediction_error)}\n"
    print("Percentual Errors Greater than 5%: ", percentual_errors_greater_than_005)
    with open("error_log", "a") as file: file.write(error_str)

    true_objective = np.vstack(converged_evaluated_prediction_elites['objective'])
    predicted_objective = np.vstack(converged_unevaluated_prediction_elites['objective'])
    pprint(predicted_objective, true_objective, percentual_error)
    print("\nMAE Error: ", mae_error, "\n", "MSE Error: ", mse_error, "\n", "Mean Percentual Error: ", mean_percentual_error, "\n")

    os.makedirs("csv") if not os.path.exists("csv") else None
    subprocess.run("mv *.csv csv", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    gc.collect()
    return
    

if __name__ == "__main__":

    subprocess.run("clean", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) # custom bash command to remove all files from last run
    exec_start = time.time()

    for i in range(TEST_RUNS):

        gc.collect()

        benchmark_domains = []
        
        sail(initial_seed=i, sail_custom_flag=True, pred_verific_flag=True,  hybrid_flag=True)
        sail(initial_seed=i, sail_custom_flag=True, pred_verific_flag=False, hybrid_flag=True)
        sail(initial_seed=i, sail_custom_flag=True, pred_verific_flag=True,  greedy_flag=True)
        sail(initial_seed=i, sail_custom_flag=True, pred_verific_flag=False, greedy_flag=True)
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
    subprocess.run("find . -type f -name '*_heatmaps.mp4' -exec mv -i '{}' . \;", shell=True)
    subprocess.run(f"cp config/config.ini {timestamp}/reproduction_info.txt", shell=True)
    subprocess.run(f'mv csv *.csv *.mp4 imgs error_log {timestamp}', shell=True)
    if not os.path.exists("benchmarks"): os.makedirs("benchmarks")
    subprocess.run(f"mv {timestamp} benchmarks", shell=True)

    subprocess.run(f"rm *.log *.dat", shell=True)

    exec_end = time.time()
    exec_time = exec_end - exec_start
    print("\nExecution time (minutes): " + str(exec_time/60))
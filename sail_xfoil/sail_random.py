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
from xfoil.simulate_airfoils_singleprocess import xfoil
from xfoil.generate_airfoils_singleprocess import generate_parsec_coordinates
from acq_functions.acq_ucb import acq_ucb
from gp.initialize_archive import initialize_archive
from gp.fit_gp_model import fit_gp_model
from gp.predict_objective import predict_objective
from map_elites import map_elites
from utils.pprint_nd import pprint, pprint_nd
from numpy import float64

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
N_RANDOM_SEARCH_EVALS = config.ACQ_N_OBJ_EVALS
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


def sail(initial_seed):

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

    pred_archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = -1,
        seed=seed,
        )

    obj_archive, init_solutions, init_obj_evals = initialize_archive(obj_archive, seed)
    subprocess.run(f"rm *.log *.dat", shell=True)

    sol_array = np.array(init_solutions)
    obj_array = np.array(init_obj_evals)

    print("\n ## Exit Initialization ##")
    print(" ## Enter Random Search Loop ##\n\n")

    eval_budget = N_RANDOM_SEARCH_EVALS

    while(eval_budget >= BATCH_SIZE):

        ranges = np.array(SOL_VALUE_RANGE)

        def uniform_sample():
            uniform_sample = np.random.uniform(ranges[:, 0], ranges[:, 1], SOL_DIMENSION)
            return uniform_sample

        random_samples = np.array([uniform_sample() for i in range(BATCH_SIZE)])

        generate_parsec_coordinates(random_samples)
        convergence_errors, success_indices, obj_batch = xfoil(iterations=BATCH_SIZE)

        converged_samples = random_samples[success_indices]

        pprint(random_samples)
        random_samples_behavior = random_samples[:, 1:3]     

        pprint(random_samples_behavior)

        converged_behavior = random_samples_behavior[success_indices]

        obj_archive.add(converged_samples, obj_batch, converged_behavior)

        sol_array = np.vstack((sol_array, converged_samples)) # dtype=float64
        obj_array = np.vstack((obj_array, obj_batch.reshape(-1,1))) # dtype=float64

        eval_budget -= BATCH_SIZE

        pprint(obj_batch)
        print("\n\nAirfoil Convergence Errors: " + str(convergence_errors))
        print("Remaining Random Search Obj Evals: " + str(eval_budget) + "\n\n")

    print("\n\n ## Exit Random Search Loop ##")
    print(" ## Enter Prediction Loop ##\n\n")

    gp_model = fit_gp_model(sol_array, obj_array)

    pred_emitter = [
        GaussianEmitter(
        archive=obj_archive,
        sigma=0.5,
        bounds=SOL_VALUE_RANGE,
        batch_size=BATCH_SIZE,
        initial_solutions=[elite.solution for elite in obj_archive],
        seed=seed
    )]

    pred_archive = map_elites(pred_archive, pred_emitter, gp_model, PRED_N_EVALS, predict_objective)

    print("[...] Terminate sail()")
    gc.collect()

    return obj_archive, pred_archive


def verify_prediction_archive(pred_dataframe):
        
        verified_obj_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = -1
        )

        unverified_obj_archive = GridArchive( # subset of prediction archive that only contains converged solutions
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = -1
        )

        pred_error_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600,
            threshold_min = -1
        )

        elites = pred_dataframe.loc[:, "solution_0":"solution_10"].to_numpy()
        behavior = pred_dataframe.loc[:, "measure_0":"measure_1"].to_numpy()

        pred_obj = pred_dataframe.loc[:, "objective"].to_numpy()

        n_invalid_elites = 0
        converged_samples = np.empty((0, SOL_DIMENSION), dtype=float64)
        converged_behavior = np.empty((0, BHV_DIMENSION), dtype=float64)
        converged_obj = np.empty(0, dtype=float64)
        converged_pred_obj = np.empty(0, dtype=float64)


        for i in range(0, elites.shape[0], BATCH_SIZE):

            try:
                elite_batch = elites[i:i+BATCH_SIZE]
            except:
                elite_batch = elites[i:]

            valid_indices = generate_parsec_coordinates(elite_batch)

            if valid_indices.size != elite_batch.shape[0]:
                print("\n\nwtf? intersecting polynomials during verification? this shouldnt happen\n\n")

            convergence_errors, success_indices, true_obj_batch = xfoil(iterations=elite_batch.shape[0])
          
            n_invalid_elites += convergence_errors
            success_indices_outside = np.array(success_indices, dtype=int) + i

            converged_behavior = np.append(converged_behavior, behavior[success_indices_outside], axis=0)
            converged_samples = np.append(converged_samples, elite_batch[success_indices], axis=0)
            converged_obj = np.append(converged_obj, true_obj_batch, axis=0)
            converged_pred_obj = np.append(converged_pred_obj, pred_obj[success_indices_outside], axis=0)
        
        pred_error = (converged_obj - converged_pred_obj) ** 2
        
        pprint(converged_samples)
        pprint(converged_obj)
        pprint(converged_pred_obj)
        pprint(pred_error)

        perc_invalid_elites = (n_invalid_elites / elites.shape[0]) * 100
        print("\nNumber of invalid elites: " + str(n_invalid_elites))
        print("Percentage of invalid elites: " + str(round(perc_invalid_elites, 2)) + "\n\n")


        verified_obj_archive.add(converged_samples, converged_obj, converged_behavior)
        unverified_obj_archive.add(converged_samples, converged_pred_obj, converged_behavior)
        pred_error_archive.add(converged_samples, pred_error, converged_behavior)

        return verified_obj_archive, unverified_obj_archive, pred_error_archive, perc_invalid_elites


if __name__ == "__main__":

    exec_start = time.time()

    mse_array = np.empty(0, dtype=float64)
    qd_score_array = np.empty(0, dtype=float64) # referring to verified_obj_archive

    for i in range(TEST_RUNS):
        data = {}

        obj_archive, pred_archive = sail(i)

        obj_dataframe = obj_archive.as_pandas(include_solutions=True)
        obj_dataframe.to_csv(f"obj_archive_{i}.csv", index=False)

        pred_dataframe = pred_archive.as_pandas(include_solutions=True)
        pred_dataframe.to_csv(f"pred_archive_{i}.csv", index=False)

        verified_obj_archive, unverified_obj_archive, pred_error_archive, perc_invalid_elites = verify_prediction_archive(pred_dataframe)

        verified_obj_dataframe = verified_obj_archive.as_pandas(include_solutions=True)
        verified_obj_dataframe.to_csv(f"verified_obj_archive_{i}.csv", index=False)

        unverified_obj_dataframe = verified_obj_archive.as_pandas(include_solutions=True)
        unverified_obj_dataframe.to_csv(f"unverified_obj_archive_{i}.csv", index=False)

        pred_error_dataframe = pred_error_archive.as_pandas(include_solutions=True)
        pred_error_dataframe.to_csv(f"pred_error_archive_{i}.csv", index=False)

        mse = np.mean(pred_error_dataframe["objective"])
        mse_array = np.append(mse_array, mse)

        qd_score = np.sum(verified_obj_dataframe.loc[:, "objective"])
        qd_score_array = np.append(qd_score_array, qd_score)

        # copy files to windows directory for usage in Rstudio
        subprocess.run(f"cp obj_archive_{i}.csv pred_archive_{i}.csv verified_obj_archive_{i}.csv pred_error_archive_{i}.csv unverified_obj_archive_{i}.csv /mnt/c/Users/patri/Desktop/Thesis/archives_xfoil", shell=True, check=True)

        # pack all data from the current iteration into one dictionary
        run_data = {
            "pred_dataframe": pred_dataframe.to_json(orient="split"),
            "verified_obj_dataframe": verified_obj_dataframe.to_json(orient="split"),
            "unverified_obj_dataframe": unverified_obj_dataframe.to_json(orient="split"),
            "pred_error_dataframe": pred_error_dataframe.to_json(orient="split"),
            "mse": mse,
            "perc_invalid_elites": perc_invalid_elites,
            "qd_score": qd_score
        }

        data[f"run"] = run_data

        with open(f"run_data_{i}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

        # with open(f"classifier_data_{i}.json", "w") as json_file:
        #     json.dump(classifier_data, json_file, indent=4)

        run_data = {}
        data = {}
        obj_archive.clear()
        pred_archive.clear()
        verified_obj_archive.clear()
        unverified_obj_archive.clear()
        pred_error_archive.clear()

        pprint(mse_array)
        pprint(qd_score_array)

    subprocess.run(f"rm *.log *.dat", shell=True)
    subprocess.run(f"mv *.csv csv", shell=True)

    exec_end = time.time()
    exec_time = exec_end - exec_start
    print("\nExecution time (minutes): " + str(exec_time/60))
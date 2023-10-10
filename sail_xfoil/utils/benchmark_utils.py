###### Archive packages #####
from ribs.archives import GridArchive
from numpy import float64
import numpy as np
import subprocess
import json
import gc
import pandas

###### Import Custom Scripts ######
from xfoil.simulate_airfoils import xfoil
from xfoil.generate_airfoils import generate_parsec_coordinates
from acq_functions.acq_ucb import acq_ucb
from utils.pprint_nd import pprint, pprint_fstring

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


def store_benchmark_data(i, obj_archive, pred_archive, sail_vanilla_flag=False, sail_custom_flag=False, sail_random_flag=False, pred_verific_flag=False):
        
        data = {}

        if sail_vanilla_flag:
            domain = "vanilla"
        if sail_custom_flag:
            domain = "custom"
        if sail_random_flag:
            domain = "random"
        if pred_verific_flag:
            domain = domain + "_prediction_verification"

        print(domain)

        obj_dataframe = obj_archive.as_pandas(include_solutions=True)
        obj_dataframe.to_csv(f"obj_archive_{domain}_{i}.csv", index=False)

        pred_dataframe = pred_archive.as_pandas(include_solutions=True)
        pred_dataframe.to_csv(f"pred_archive_{domain}_{i}.csv", index=False)

        verified_obj_archive, unverified_obj_archive, pred_error_archive, perc_invalid_elites, df_obj_vs_pred_vs_error = verify_prediction_archive(pred_dataframe)
        
        df_obj_vs_pred_vs_error.to_csv(f"obj_vs_pred_vs_error_{domain}_{i}.csv", index=False)

        verified_obj_dataframe = verified_obj_archive.as_pandas(include_solutions=True)
        verified_obj_dataframe.to_csv(f"verified_obj_archive_{domain}_{i}.csv", index=False)

        unverified_obj_dataframe = verified_obj_archive.as_pandas(include_solutions=True)
        unverified_obj_dataframe.to_csv(f"unverified_obj_archive_{domain}_{i}.csv", index=False)

        pred_error_dataframe = pred_error_archive.as_pandas(include_solutions=True)
        pred_error_dataframe.to_csv(f"pred_error_archive_{domain}_{i}.csv", index=False)

        mse = np.mean(pred_error_dataframe["objective"])

        qd_score = np.sum(verified_obj_dataframe.loc[:, "objective"])

        # copy files to windows directory for usage in Rstudio
        subprocess.run(f"cp obj_archive_{domain}_{i}.csv pred_archive_{domain}_{i}.csv verified_obj_archive_{domain}_{i}.csv pred_error_archive_{domain}_{i}.csv unverified_obj_archive_{domain}_{i}.csv /mnt/c/Users/patri/Desktop/Thesis/archives_xfoil", shell=True, check=True)

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

        data[f"run_{domain}_{i}"] = run_data

        with open(f"run_data_{domain}_{i}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)

        run_data = {}
        data = {}
        obj_archive.clear()
        pred_archive.clear()
        verified_obj_archive.clear()
        unverified_obj_archive.clear()
        pred_error_archive.clear()

        gc.collect()

        return mse, qd_score, perc_invalid_elites


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
        converged_elites = np.empty((0, SOL_DIMENSION), dtype=float64)
        converged_behavior = np.empty((0, BHV_DIMENSION), dtype=float64)
        converged_obj = np.empty(0, dtype=float64)
        converged_pred_obj = np.empty(0, dtype=float64)


        for i in range(0, elites.shape[0], BATCH_SIZE):

            try:
                elite_batch = elites[i:i+BATCH_SIZE]
            except:
                elite_batch = elites[i:]

            valid_indices, surface_batch = generate_parsec_coordinates(elite_batch)

            if valid_indices.size != elite_batch.shape[0]:
                print("\n\nwtf? intersecting polynomials during verification? this shouldnt happen\n\n")

            convergence_errors, success_indices, true_obj_batch = xfoil(iterations=elite_batch.shape[0])
          
            n_invalid_elites += convergence_errors
            success_indices_outside = np.array(success_indices, dtype=int) + i

            converged_behavior = np.append(converged_behavior, behavior[success_indices_outside], axis=0)
            converged_elites = np.append(converged_elites, elite_batch[success_indices], axis=0)
            converged_obj = np.append(converged_obj, true_obj_batch, axis=0)
            converged_pred_obj = np.append(converged_pred_obj, pred_obj[success_indices_outside], axis=0)
        
        pred_error = (converged_obj - converged_pred_obj) ** 2
        
        pprint(converged_elites)
        pprint_fstring(converged_obj, converged_pred_obj, pred_error)

        df_obj_vs_pred_vs_error = pandas.DataFrame({"converged_obj": converged_obj, "converged_pred_obj": converged_pred_obj, "pred_error": pred_error})
        # ToDo: paste solution columns to df above     "solution": converged_elites
        
        perc_invalid_elites = (n_invalid_elites / elites.shape[0]) * 100
        print("\nNumber of invalid elites: " + str(n_invalid_elites))
        print("Percentage of invalid elites: " + str(round(perc_invalid_elites, 2)) + "\n\n")


        verified_obj_archive.add(converged_elites, converged_obj, converged_behavior)
        unverified_obj_archive.add(converged_elites, converged_pred_obj, converged_behavior)
        pred_error_archive.add(converged_elites, pred_error, converged_behavior)

        gc.collect()

        return verified_obj_archive, unverified_obj_archive, pred_error_archive, perc_invalid_elites, df_obj_vs_pred_vs_error
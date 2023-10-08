###### Archive packages #####
from ribs.archives import GridArchive
from numpy import float64
import numpy as np
import subprocess
from tqdm import tqdm
import json
import gc

###### Import Custom Scripts ######
from xfoil.simulate_airfoils import xfoil
from xfoil.generate_airfoils import generate_parsec_coordinates
from acq_functions.acq_ucb import acq_ucb
from gp.fit_gp_model import fit_gp_model
from map_elites import map_elites
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


def sail_custom(acq_archive, obj_archive, gp_model, acq_emitter, sol_array, obj_array, eval_budget):
    while(eval_budget >= BATCH_SIZE):

        # update acquisition values
        acq_archive = store_n_best_elites(obj_archive, obj_archive.stats.num_elites, update_acq=True, gp_model=gp_model)

        # evolve acquisition archive until minimum of 10 elites has been found
        acq_archive, new_elite_archive = map_elites(acq_archive, acq_emitter, gp_model, ACQ_N_MAP_EVALS, acq_ucb)
        while new_elite_archive.stats.num_elites < BATCH_SIZE:
            print("\nElites in New Elite Archive: " + str(new_elite_archive.stats.num_elites))
            print("Enter second acquisition loop iteration")
            acq_archive, new_elite_archive = map_elites(acq_archive, acq_emitter, gp_model, ACQ_N_MAP_EVALS, acq_ucb, new_elite_archive=new_elite_archive)

        # select & evaluate acquisition elites
        #acq_elite_batch = new_elite_archive.sample_elites(BATCH_SIZE)
        new_elite_batch = sorted(new_elite_archive, key=lambda x: x.objective, reverse=True)[:10]

        new_elite_solutions = np.array([elite.solution for elite in new_elite_batch])
        new_elite_acquisition = np.array([elite.objective for elite in new_elite_batch])
        new_elite_measures = np.array([elite.measures for elite in new_elite_batch])

        # acq_archive only contains valid solutions (= non-intersecting polynomials), therefore no need to check for validity using valid_indices
        valid_indices, surface_area_batch = generate_parsec_coordinates(new_elite_solutions)
        convergence_errors, success_indices, new_elites_objectives = xfoil(BATCH_SIZE)

        # store evaluations for GP model
        sol_array = np.vstack((sol_array, new_elite_solutions[success_indices])) # dtype=float64
        obj_array = np.vstack((obj_array, new_elites_objectives.reshape(-1,1))) # dtype=float64

        eval_budget -= BATCH_SIZE

        new_elite_acquisition = new_elite_acquisition[success_indices]
        pprint_fstring(new_elite_acquisition, new_elites_objectives)
        print("\nElites in New Elite Archive: " + str(new_elite_archive.stats.num_elites))
        print("Acq Archive Size: " + str(acq_archive.stats.num_elites)) # print with only four decimals
        print("Acq QD (Custom): " +  str(int(acq_archive.stats.qd_score)))
        print("Airfoil Convergence Errors: " + str(convergence_errors))
        print("Obj Elites (before): " + str(obj_archive.stats.num_elites))
        obj_archive.add(new_elite_solutions[success_indices], new_elites_objectives, new_elite_measures[success_indices])
        print("Obj Elites (after): " + str(obj_archive.stats.num_elites))
        print("Remaining ACQ Precise Evals: " + str(eval_budget) + "\n\n")

        gp_model = fit_gp_model(sol_array, obj_array)

    return obj_archive, gp_model
    

def sail_vanilla(acq_archive, obj_archive, gp_model, acq_emitter, sol_array, obj_array, eval_budget):
    while(eval_budget >= BATCH_SIZE):
        eval_budget -= BATCH_SIZE

        acq_archive = store_n_best_elites(obj_archive, obj_archive.stats.num_elites, update_acq=True, gp_model=gp_model)    # update acquisition values
        acq_archive, _ = map_elites(acq_archive, acq_emitter, gp_model, ACQ_N_MAP_EVALS, acq_ucb)                           # evolve acquisition archive

        acq_elite_batch = acq_archive.sample_elites(BATCH_SIZE)        
        acq_elite_solutions = acq_elite_batch[0]
        acq_elite_acquisitions = acq_elite_batch[1]
        acq_elite_measures = acq_elite_batch[2]

        # acq_archive only contains valid solutions (= non-intersecting polynomials), therefore no need to check for validity using valid_indices
        _, surface_area_batch = generate_parsec_coordinates(acq_elite_solutions)
        convergence_errors, success_indices, acq_elite_objectives = xfoil(BATCH_SIZE)

        # store evaluations for GP model
        sol_array = np.vstack((sol_array, acq_elite_solutions[success_indices])) # dtype=float64
        obj_array = np.vstack((obj_array, acq_elite_objectives.reshape(-1,1))) # dtype=float64

        acquisitions = acq_elite_acquisitions[success_indices]
        pprint_fstring(acquisitions, acq_elite_objectives)
        print("Acq Archive Size: " + str(acq_archive.stats.num_elites))
        print("Acq QD (Vanilla): " +  str(int(acq_archive.stats.qd_score)))
        print("Airfoil Convergence Errors: " + str(convergence_errors))
        print("Obj Elites (before): " + str(obj_archive.stats.num_elites))
        obj_archive.add(acq_elite_solutions[success_indices], acq_elite_objectives, acq_elite_measures[success_indices])
        print("Obj Elites (after): " + str(obj_archive.stats.num_elites))
        print("Remaining ACQ Precise Evals: " + str(eval_budget) + "\n\n")

        gp_model = fit_gp_model(sol_array, obj_array)

        return obj_archive, gp_model

def sail_random(acq_archive, obj_archive, gp_model, acq_emitter, sol_array, obj_array, eval_budget):
    while(eval_budget >= BATCH_SIZE):

        ranges = np.array(SOL_VALUE_RANGE)

        def uniform_sample():
            uniform_sample = np.random.uniform(ranges[:, 0], ranges[:, 1], SOL_DIMENSION)
            return uniform_sample

        random_samples = np.array([uniform_sample() for i in range(BATCH_SIZE)])

        generate_parsec_coordinates(random_samples)
        convergence_errors, success_indices, obj_batch = xfoil(iterations=BATCH_SIZE)

        converged_samples = random_samples[success_indices]
        converged_behavior = random_samples[success_indices, 1:3]

        obj_archive.add(converged_samples, obj_batch, converged_behavior)

        sol_array = np.vstack((sol_array, converged_samples)) # dtype=float64
        obj_array = np.vstack((obj_array, obj_batch.reshape(-1,1))) # dtype=float64

        eval_budget -= BATCH_SIZE

        pprint(obj_batch)
        print("\n\nAirfoil Convergence Errors: " + str(convergence_errors))
        print("Remaining Random Search Obj Evals: " + str(eval_budget) + "\n\n")

        return obj_archive, gp_model


def store_n_best_elites(archive, n, update_acq=True, gp_model=None):

    n_elites = sorted(archive, key=lambda x: x.objective, reverse=True)[:n]

    if update_acq:
        n_elite_acq = acq_ucb(np.array([elite.solution for elite in n_elites]), gp_model)
    else:
        n_elite_acq = [elite.objective for elite in n_elites]

    archive.clear()
    archive.add([elite.solution for elite in n_elites], n_elite_acq, [elite.measures for elite in n_elites])

    return archive


def store_benchmark_data(i, obj_archive, pred_archive, sail_vanilla_flag=False, sail_custom_flag=False, sail_random_flag=False):
        
        data = {}

        if sail_vanilla_flag:
            domain = "vanilla"
        if sail_custom_flag:
            domain = "custom"
        if sail_random_flag:
            domain = "random"

        obj_dataframe = obj_archive.as_pandas(include_solutions=True)
        obj_dataframe.to_csv(f"obj_archive_{domain}_{i}.csv", index=False)

        pred_dataframe = pred_archive.as_pandas(include_solutions=True)
        pred_dataframe.to_csv(f"pred_archive_{domain}_{i}.csv", index=False)

        verified_obj_archive, unverified_obj_archive, pred_error_archive, perc_invalid_elites = verify_prediction_archive(pred_dataframe)

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

            valid_indices, _ = generate_parsec_coordinates(elite_batch)

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
        pprint(converged_obj)
        pprint(converged_pred_obj)
        pprint(pred_error)

        perc_invalid_elites = (n_invalid_elites / elites.shape[0]) * 100
        print("\nNumber of invalid elites: " + str(n_invalid_elites))
        print("Percentage of invalid elites: " + str(round(perc_invalid_elites, 2)) + "\n\n")


        verified_obj_archive.add(converged_elites, converged_obj, converged_behavior)
        unverified_obj_archive.add(converged_elites, converged_pred_obj, converged_behavior)
        pred_error_archive.add(converged_elites, pred_error, converged_behavior)

        gc.collect()

        return verified_obj_archive, unverified_obj_archive, pred_error_archive, perc_invalid_elites
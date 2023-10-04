###### Archive packages #####
import numpy as np
import subprocess
import json
from ribs.emitters import GaussianEmitter
from ribs.archives import GridArchive


###### Import Custom Scripts ######
from utils.simulate_robotarm import simulate_obj, simulate_bhv
from acq_functions.acq_ucb import acq_ucb
from example.example_functions import example_objective_function, example_behavior_function, example_variation_function
from utils.initialize_archive import initialize_archive
from utils.fit_gp_model import fit_gp_model
from utils.pprint import pprint
from utils.predict_objective import predict_objective
from map_elites import map_elites
from numpy import float64

###### Configurable Variables ######
from config import Config
config = Config('config.ini')
ACQ_N_OBJ_EVALS = config.ACQ_N_OBJ_EVALS
ACQ_N_MAP_EVALS = config.ACQ_N_MAP_EVALS
PRED_N_EVALS = config.PRED_N_EVALS
PARALLEL_BATCH_SIZE = config.PARALLEL_BATCH_SIZE
SOL_DIMENSION = config.SOL_DIMENSION
BHV_ARCHIVE_DIMENSION = config.BHV_ARCHIVE_DIMENSION
BHV_VALUE_RANGE = config.BHV_VALUE_RANGE
BHV_NUMBER_BINS = config.BHV_NUMBER_BINS
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE

def sail(initial_seed):

    print("Initialize sail() [...]")

    obj_archive = GridArchive(
        solution_dim=SOL_DIMENSION,         # Dimension of solution vector
        dims=BHV_NUMBER_BINS,               # Discretization of behavioral bins
        ranges=BHV_VALUE_RANGE,             # Possible values for behavior vector
        qd_score_offset=-600)
    
    acq_archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600)
    
    pred_archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600
    )

    obj_archive, init_solutions, init_obj_evals, init_bhv_evals = initialize_archive(obj_archive, simulate_obj, simulate_bhv, initial_seed)

    obj_archive.add(init_solutions, init_obj_evals.ravel(), init_bhv_evals)

    sol_array = []
    obj_array = []

    sol_array = init_solutions
    obj_array = init_obj_evals

    gp_model = fit_gp_model(sol_array, obj_array)

    acq_emitter = [
        GaussianEmitter(
        archive=acq_archive,
        sigma=0.5,
        bounds= np.array(SOL_VALUE_RANGE),
        batch_size=PARALLEL_BATCH_SIZE,
        initial_solutions=init_solutions,
    )]

    print(" ## Exit Initialization ##")
    print(" ## Enter Acquisition Loop ##\n\n")

    eval_budget = ACQ_N_OBJ_EVALS

    while(eval_budget-PARALLEL_BATCH_SIZE > 0): # future addition: add threshhold condition for predictive performance of the model
        
        print("Remaining ACQ Evals: " + str(eval_budget))

        acq_archive.clear()

        acq_archive = map_elites(acq_archive, acq_emitter, gp_model, ACQ_N_MAP_EVALS, acq_ucb, simulate_bhv)
        
        acq_elites = acq_archive.sample_elites(PARALLEL_BATCH_SIZE)     # Select acquisition elites (sobol sample in original paper)
        obj_evals = simulate_obj(acq_elites[0])
        
        obj_archive.add(acq_elites[0], obj_evals.ravel(), acq_elites[2])

        eval_budget -= PARALLEL_BATCH_SIZE

        sol_array = np.vstack((sol_array, acq_elites[0]), dtype=float64)
        obj_array = np.vstack((obj_array, obj_evals.reshape(-1,1)), dtype=float64)

        gp_model = fit_gp_model(sol_array, obj_array)

    print(" ## Exit Acquisition Loop ##")
    print(" ## Enter Prediction Loop ##")

    obj_emitter = [
        GaussianEmitter(
        archive=obj_archive,
        sigma=0.5,
        bounds=SOL_VALUE_RANGE,
        batch_size=PARALLEL_BATCH_SIZE,
        initial_solutions=sol_array,
    )]

    pred_archive = map_elites(pred_archive, obj_emitter, gp_model, PRED_N_EVALS, predict_objective, simulate_bhv)

    print("[...] Terminate sail()")

    return obj_archive, acq_archive, pred_archive


def verify_prediction_archive(pred_dataframe):
        
        verified_obj_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600
        )

        pred_error_archive = GridArchive(
            solution_dim=SOL_DIMENSION,
            dims=BHV_NUMBER_BINS,
            ranges=BHV_VALUE_RANGE,
            qd_score_offset=-600
        )

        elites = pred_dataframe.loc[:, "solution_0":"solution_11"].to_numpy()
        behavior = pred_dataframe.loc[:, "measure_0":"measure_1"].to_numpy()

        pred_obj = pred_dataframe.loc[:, "objective"].to_numpy()
        true_obj = simulate_obj(elites)

        pred_error = (true_obj - pred_obj) ** 2

        verified_obj_archive.add(elites, true_obj, behavior)
        pred_error_archive.add(elites, pred_error, behavior)

        return verified_obj_archive, pred_error_archive


if __name__ == "__main__":

    data = {}
    mse_array = np.empty(0, dtype=float64)
    qd_score_array = np.empty(0, dtype=float64) # referring to verified_obj_archive

    for i in range(2):

        obj_archive, acq_archive, pred_archive = sail(i)

        pred_dataframe = pred_archive.as_pandas(include_solutions=True)
        pred_dataframe.to_csv(f"pred_archive_{i}.csv", index=False)

        verified_obj_archive, pred_error_archive = verify_prediction_archive(pred_dataframe)

        verified_obj_dataframe = verified_obj_archive.as_pandas(include_solutions=True)
        verified_obj_dataframe.to_csv(f"verified_obj_archive_{i}.csv", index=False)

        pred_error_dataframe = pred_error_archive.as_pandas(include_solutions=True)
        pred_error_dataframe.to_csv(f"pred_error_archive_{i}.csv", index=False)

        mse = np.mean(pred_error_dataframe["objective"])
        mse_array = np.append(mse_array, mse)

        qd_score = np.sum(verified_obj_dataframe.loc[:, "objective"])
        qd_score_array = np.append(qd_score_array, qd_score)

        # copy files to windows directory for usage in Rstudio
        subprocess.run(f"cp pred_archive_{i}.csv verified_obj_archive_{i}.csv pred_error_archive_{i}.csv /mnt/c/Users/patri/Desktop/Thesis/archives_robotarm", shell=True, check=True)

        # pack all data from the current iteration into one dictionary
        run_data = {
            "pred_dataframe": pred_dataframe.to_json(orient="split"),
            "verified_obj_dataframe": verified_obj_dataframe.to_json(orient="split"),
            "pred_error_dataframe": pred_error_dataframe.to_json(orient="split"),
            "mse": mse,
            "qd_score": qd_score
        }

        data[f"run_{i}"] = run_data

    with open("results.json", "w") as json_file:
        json.dump(data, json_file, indent=4)
    
    pprint(mse_array)
    pprint(qd_score_array)

    subprocess.run(f"rm *.csv", shell=True, check=True)
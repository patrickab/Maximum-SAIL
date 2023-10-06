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
from xfoil.simulate_airfoils import xfoil
from xfoil.generate_airfoils import generate_parsec_coordinates
from acq_functions.acq_ucb import acq_ucb
from gp.initialize_archive import initialize_archive
from gp.fit_gp_model import fit_gp_model
from gp.predict_objective import predict_objective
from map_elites import map_elites
from utils.pprint_nd import pprint, pprint_fstring
from numpy import float64

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
    
    acq_archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = 0.2,
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

    gp_model = fit_gp_model(init_solutions, init_obj_evals)

    sol_array = np.array(init_solutions)
    obj_array = np.array(init_obj_evals)

    obj_elites = np.array([elite.solution for elite in obj_archive])
    obj_elites_acq = acq_ucb(obj_elites, gp_model)
    obj_elites_measures = np.array([elite.measures for elite in obj_archive])

    acq_archive.add(obj_elites, obj_elites_acq, obj_elites_measures)

    acq_emitter = [
        GaussianEmitter(
        archive=acq_archive,
        sigma=1,
        bounds= np.array(SOL_VALUE_RANGE),
        batch_size=BATCH_SIZE,
        initial_solutions=init_solutions, # these solutions are never used, as the archive is never empty - however, specification is required for initializing the GaussianEmitter class
        seed=seed
    )]

    classifier_data = {
        "converged": np.empty((0, SOL_DIMENSION), dtype=float64),
        "not_converged": np.empty((0, SOL_DIMENSION), dtype=float64),
    }

    print("\n ## Exit Initialization ##")
    print(" ## Enter Acquisition Loop ##\n\n")

    eval_budget = ACQ_N_OBJ_EVALS
    while(eval_budget >= BATCH_SIZE):

        # update acquisition values
        acq_archive = store_n_best_elites(obj_archive, obj_archive.stats.num_elites, update_acq=True, gp_model=gp_model)

        # evolve acquisition archive
        acq_archive, new_elite_archive = map_elites(acq_archive, acq_emitter, gp_model, ACQ_N_MAP_EVALS, acq_ucb)

        # select & evaluate acquisition elites
        #acq_elite_batch = new_elite_archive.sample_elites(BATCH_SIZE)
        acq_elite_batch = sorted(new_elite_archive, key=lambda x: x.objective, reverse=True)[:10]
        acq_elites = acq_elite_batch

        # acq_archive only contains valid solutions (= non-intersecting polynomials), therefore no need to check for validity using valid_indices
        valid_indices, surface_area_batch = generate_parsec_coordinates(acq_elites)
        convergence_errors, success_indices, obj_batch = xfoil(BATCH_SIZE, surface_area_batch)

        # select & store converged solutions
        converged_elites = acq_elites[success_indices]        
        obj_archive.add(converged_elites, obj_batch, acq_elite_batch[2][success_indices])

        # store evaluations for GP model
        sol_array = np.vstack((sol_array, converged_elites)) # dtype=float64
        obj_array = np.vstack((obj_array, obj_batch.reshape(-1,1))) # dtype=float64

        eval_budget -= BATCH_SIZE

        # Define the format strings
        acq_batch = acq_elite_batch[1][success_indices]
        pprint_fstring(acq_batch, obj_batch)
        print("Acq Archive Size: " + str(acq_archive.stats.num_elites))
        print("Acq QD Score: " +  str(acq_archive.stats.qd_score))
        print("Airfoil Convergence Errors: " + str(convergence_errors))
        print("Remaining ACQ Precise Evals: " + str(eval_budget) + "\n\n")

        gp_model = fit_gp_model(sol_array, obj_array)

    print("\n\n ## Exit Acquisition Loop ##")
    print(" ## Enter Prediction Loop ##\n\n")

    pred_emitter = [
        GaussianEmitter(
        archive=obj_archive,
        sigma=0.5,
        bounds=SOL_VALUE_RANGE,
        batch_size=BATCH_SIZE,
        initial_solutions=[elite.solution for elite in obj_archive],
        seed=seed
    )]

    pred_archive, new_elites = map_elites(pred_archive, pred_emitter, gp_model, PRED_N_EVALS, predict_objective)

    print("[...] Terminate sail()")
    gc.collect()

    return obj_archive, acq_archive, pred_archive, classifier_data


def collect_classifier_data(acq_elites, success_indices, classifier_data):

    for index in success_indices:
        classifier_data["converged"] = np.append(classifier_data["converged"], acq_elites[index].reshape(1,-1), axis=0)
        classifier_data["not_converged"] = np.append(classifier_data["not_converged"], acq_elites[index].reshape(1,-1), axis=0)


def store_n_best_elites(archive, n, update_acq=True, gp_model=None):

    n_elites = sorted(archive, key=lambda x: x.objective, reverse=True)[:n]

    if update_acq:
        n_elite_acq = acq_ucb(np.array([elite.solution for elite in n_elites]), gp_model)
    else:
        n_elite_acq = [elite.objective for elite in n_elites]

    archive.clear()
    archive.add([elite.solution for elite in n_elites], n_elite_acq, [elite.measures for elite in n_elites])

    return archive        


def train_classifieres(classifier_data):

    import json
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, classification_report

    warnings.filterwarnings("ignore")

    converged_elites = classifier_data["converged"]
    not_converged_elites = classifier_data["not_converged"]

    # Combine converged and not converged data
    x = np.vstack((converged_elites, not_converged_elites))
    y = np.hstack((np.ones(len(converged_elites)), np.zeros(len(not_converged_elites))))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print(y_train)
    print(y_test)
    percentage_y_0 = (len(y_train[y_train == 0]) / len(y_train)) * 100
    percentage_y_1 = (len(y_train[y_train == 1]) / len(y_train)) * 100
    print(f"Percentage of y_train == 0: {percentage_y_0}")
    print(f"Percentage of y_train == 1: {percentage_y_1}")

    # Initialize classifiers
    classifiers = {
        #"Support Vector Machine": SVC(random_state=1337),
        #"Neural Network": MLPClassifier(random_state=1337),
        "Random Forest": RandomForestClassifier(random_state=1337),
        #"Logistic Regression": LogisticRegression(random_state=1337),
        #"Naive Bayes": GaussianNB(),
        #"K-Nearest Neighbors": KNeighborsClassifier()
    }

    # Train and evaluate classifiers
    results = {}
    for classifier_name, classifier in classifiers.items():
        print(f"Training {classifier_name}...")
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        results[classifier_name] = {"Accuracy": accuracy, "Report": report}
        print(f"Accuracy: {accuracy}")
        print(report)

    print(results)

    return results


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

            valid_indices = generate_parsec_coordinates(elite_batch)

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

        return verified_obj_archive, unverified_obj_archive, pred_error_archive, perc_invalid_elites


if __name__ == "__main__":

    exec_start = time.time()

    mse_array = np.empty(0, dtype=float64)
    qd_score_array = np.empty(0, dtype=float64) # referring to verified_obj_archive

    for i in range(TEST_RUNS):
        data = {}

        obj_archive, acq_archive, pred_archive, classifier_data = sail(i)

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

        print("Verfied Obj QD Score: " +  str(acq_archive.stats.qd_score))
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
        acq_archive.clear()
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
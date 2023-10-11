###### Archive packages #####
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from botorch.models import SingleTaskGP
import numpy as np
import gc

###### Import Custom Scripts ######
from map_elites import map_elites
from xfoil.simulate_airfoils import xfoil
from xfoil.generate_airfoils import generate_parsec_coordinates
from gp.fit_gp_model import fit_gp_model
from gp.predict_objective import predict_objective
from utils.anytime_archive_visualizer import anytime_archive_visualizer
from utils.pprint_nd import pprint, pprint_fstring
from utils.utils import eval_xfoil_loop, maximize_obj_improvement, store_n_best_elites


###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
BATCH_SIZE = config.BATCH_SIZE
PRED_N_EVALS = config.PRED_N_EVALS
ACQ_N_OBJ_EVALS = config.ACQ_N_OBJ_EVALS
PRED_ELITE_REEVALS = config.PRED_ELITE_REEVALS
MAX_PRED_VERIFICATION = config.MAX_PRED_VERIFICATION


def prediction_verification_loop(pred_archive: GridArchive, obj_archive: GridArchive, 
                                 pred_emitter: GaussianEmitter, gp_model: SingleTaskGP, 
                                 sol_array: np.ndarray, obj_array: np.ndarray,
                                 initial_seed: int, benchmark_domain: str):

    print("\n\n ## Enter Prediction Verification Loop##")
    extra_evals = 0
    pred_n_evals = PRED_N_EVALS//(PRED_ELITE_REEVALS+1) # +1 because after the loop predictions with map_elites is called once more
    total_pred_evals = PRED_N_EVALS
    index_anytime_visualizer = (ACQ_N_OBJ_EVALS//BATCH_SIZE)+1

    gp_sol_array = sol_array
    gp_obj_array = obj_array

    while total_pred_evals > pred_n_evals:

        total_pred_evals -= pred_n_evals
        old_elites = np.array([(elite.solution, elite.index, elite.objective, elite.measures) for elite in obj_archive], dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])

        pred_archive, new_elite_archive = map_elites(pred_archive, pred_emitter, gp_model, pred_n_evals, predict_objective)         # predict new elites
        max_obj_improvement_elites, new_elites = maximize_obj_improvement(new_elite_archive, old_elites)                            # maximize expected objective improvement
 
        if MAX_PRED_VERIFICATION < extra_evals+len(max_obj_improvement_elites):
            Warning(f"MAX_PRED_VERIFICATION ({MAX_PRED_VERIFICATION}) exceeded. Exiting prediction verification loop.")             # Only enter Prediction Verification if MAX_PRED_VERIFICATION is not exceeded
            break
        
        print("Remaining evaluations: " + str(total_pred_evals))
        best_sample_elites = max_obj_improvement_elites[:MAX_PRED_VERIFICATION//PRED_ELITE_REEVALS]                                   # Evenly distribute Verifications among MAX_PRED_VERIFICATION
        pred_archive, extra_evals, gp_model, gp_sol_array, gp_obj_array = prediction_verification(max_obj_improvement_elites=best_sample_elites, pred_archive=pred_archive, obj_archive=obj_archive, sol_array=gp_sol_array, obj_array=gp_obj_array, extra_evals=extra_evals, gp_model=gp_model, initial_seed=initial_seed, benchmark_domain=benchmark_domain, index_anytime_visualizer=index_anytime_visualizer) # verify predictions
        index_anytime_visualizer += best_sample_elites.shape[0]//BATCH_SIZE if best_sample_elites.shape[0] % BATCH_SIZE == 0 else (best_sample_elites.shape[0]//BATCH_SIZE)+1 # update index_anytime_visualizer
    print(f"\n\nExtra evaluations (output): {extra_evals}\n\n")

    while total_pred_evals > pred_n_evals:
        total_pred_evals -= pred_n_evals
        pred_archive, new_elite_archive = map_elites(pred_archive, pred_emitter, gp_model, pred_n_evals, predict_objective)         # Predict until budget is exhausted

    new_elite_archive.clear()
    gc.collect()

    # ToDo: ensure that exactly MAX_PRED_VERIFICATIONs are always evaluated
    if extra_evals < MAX_PRED_VERIFICATION:
        print("\n\n\nMaximum Pred Verifications not reached\n\n\n")
    if extra_evals > MAX_PRED_VERIFICATION:
        print("\n\n\nMaximum Pred Verifications exceeded\n\n\n")

    return pred_archive, extra_evals                                                                                                        # communicate extra evaluations


def prediction_verification(max_obj_improvement_elites, 
                            pred_archive: GridArchive, obj_archive: GridArchive, 
                            sol_array: np.array, obj_array: np.array, 
                            extra_evals: int, gp_model: SingleTaskGP,
                            initial_seed: int, benchmark_domain: str, 
                            index_anytime_visualizer: int):
    """
    - Evaluates all new elites in the prediction archive.
    - Stores converged new elites if they present elite objectives.
    - Preserves obj_archive elites if predicted elites are not better.
    """

    print("### PRED VERIFICATION ###")
    print("n max obj improvment elites: " + str(len(max_obj_improvement_elites)))
    print("index_anytime_visualizer  " + str(index_anytime_visualizer))

    if len(max_obj_improvement_elites) == 0:
        extra_evals=extra_evals
        return pred_archive, extra_evals, gp_model
    
    max_improvement_sol = np.vstack(max_obj_improvement_elites['solution'])
    max_improvement_bhv = np.vstack(max_obj_improvement_elites['behavior'])
    gp_sol_array = sol_array
    gp_obj_array = obj_array

    print("Extra Evals: " + str(extra_evals))
    # evaluate in for loop to ensure BATCH_SIZE is not exceeded
    # Second line specified anytime_archive_visualizer arguments:    executed inside eval_xfoil_loop, for visualization of current state of archive in each iteration
    conv_sol, conv_obj, conv_bhv, obj_archive, extra_evals = eval_xfoil_loop(max_improvement_sol, max_improvement_bhv,
        archive=obj_archive, benchmark_domain=benchmark_domain, initial_seed=initial_seed, index_anytime_visualizer=index_anytime_visualizer, extra_evals=extra_evals)
    print("Extra Evals: " + str(extra_evals))


    # update acq archive with new elite solutions
    print(f'Pred archive size (before): {pred_archive.stats.num_elites}')
    obj_elite_sol = np.array([elite.solution for elite in obj_archive])
    obj_elite_obj = np.array([elite.objective for elite in obj_archive])
    obj_elite_bhv = np.array([elite.measures for elite in obj_archive])
    pred_elite_sol = np.array([elite.solution for elite in pred_archive])
    pred_elite_obj = np.array([elite.objective for elite in pred_archive])
    pred_elite_bhv = np.array([elite.measures for elite in pred_archive])
    converged_sol = conv_sol 
    converged_obj = conv_obj # redefine for better readability
    converged_bhv = conv_bhv
    candidate_elite_sol = np.vstack((obj_elite_sol, converged_sol, pred_elite_sol))
    candidate_elite_obj = np.hstack((obj_elite_obj, converged_obj, pred_elite_obj))
    candidate_elite_bhv = np.vstack((obj_elite_bhv, converged_bhv, pred_elite_bhv))
    pred_archive.clear()
    status_vector, _ = pred_archive.add(candidate_elite_sol, candidate_elite_obj, candidate_elite_bhv)

    print(f'Pred archive size  (after): {pred_archive.stats.num_elites}')

    print("Obj Array Shape: " + str(obj_array.shape))
    pprint_fstring(obj_array)
    # store evaluations for GP model
    gp_sol_array = np.vstack((gp_sol_array, conv_sol))
    gp_obj_array = np.hstack((gp_obj_array, conv_obj))

    gp_model = fit_gp_model(sol_array, obj_array)                                                                                       # update GP model

    return pred_archive, extra_evals, gp_model, gp_sol_array, gp_obj_array
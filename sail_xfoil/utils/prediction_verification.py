###### Archive packages #####
import numpy as np
import gc

###### Import Custom Scripts ######
from xfoil.simulate_airfoils import xfoil
from xfoil.generate_airfoils import generate_parsec_coordinates
from gp.fit_gp_model import fit_gp_model
from gp.predict_objective import predict_objective
from utils.pprint_nd import pprint, pprint_fstring
from utils.utils import eval_xfoil_loop
from map_elites import map_elites

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
BATCH_SIZE = config.BATCH_SIZE
PRED_N_EVALS = config.PRED_N_EVALS
PRED_ELITE_REEVALS = config.PRED_ELITE_REEVALS
MAX_PREDICTION_VERIFICATION = config.MAX_PREDICTION_VERIFICATION


def prediction_verification_loop(pred_archive, obj_archive, pred_emitter, gp_model, sol_array, obj_array, extra_evals=0):

    print("\n\n ## Enter Prediction Verification Loop##")
    extra_evals = 0
    pred_n_evals = PRED_N_EVALS//PRED_ELITE_REEVALS


    for i in range(PRED_ELITE_REEVALS):

        pred_archive, new_elite_archive = map_elites(pred_archive, pred_emitter, gp_model, pred_n_evals, predict_objective)                 # predict new elites
        if MAX_PREDICTION_VERIFICATION > extra_evals+new_elite_archive.stats.num_elites:
            Warning(f"MAX_PREDICTION_VERIFICATION ({MAX_PREDICTION_VERIFICATION}) exceeded. Exiting prediction verification loop.")
            break

        pred_archive, sol_array, obj_array = prediction_verification(new_elite_archive, pred_archive, obj_archive, sol_array, obj_array)    # verify predictions
        gp_model = fit_gp_model(sol_array, obj_array)                                                                                       # update GP model
        extra_evals += new_elite_archive.stats.num_elites                                                                                   # count extra evaluations
    print(f"\n\nExtra evaluations (output): {extra_evals}\n\n")

    new_elite_archive.clear()
    gc.collect()

    return pred_archive, extra_evals                                                                                                        # communicate extra evaluations


def prediction_verification(new_elite_archive, pred_archive, obj_archive, sol_array, obj_array):
    """
    - Evaluates all new elites in the prediction archive.
    - Stores converged new elites if they present elite objectives.
    - Preserves obj_archive elites if predicted elites are not better.
    """

    new_elites = np.array(
        [(elite.solution, elite.index, elite.objective, elite.measures) for elite in new_elite_archive], 
        dtype=[('solution', object), ('index', int), ('prediction', float), ('behavior', object)])

    print("New Elites: " + str(new_elite_archive.stats.num_elites))
    print(new_elites)

    new_elite_sol = np.vstack(new_elites['solution'])
    new_elite_bhv = np.vstack(new_elites['behavior'])

    conv_sol, conv_obj, conv_bhv = eval_xfoil_loop(new_elite_sol, new_elite_bhv)     # evaluate in for loop to ensure BATCH_SIZE is not exceeded

    obj_elite_sol = np.array([elite.solution for elite in obj_archive])
    obj_elite_obj = np.array([elite.objective for elite in obj_archive])
    obj_elite_bhv = np.array([elite.measures for elite in obj_archive])

    condidate_elite_sol = np.concatenate((obj_elite_sol, conv_sol))
    condidate_elite_obj = np.concatenate((obj_elite_obj, conv_obj))
    condidate_elite_bhv = np.concatenate((obj_elite_bhv, conv_bhv))

    pred_archive.clear()
    pred_archive.add(condidate_elite_sol, condidate_elite_obj, condidate_elite_bhv)

    # store evaluations for GP model
    sol_array = np.vstack((sol_array, conv_sol))
    obj_array = np.vstack((obj_array, conv_obj.reshape(-1,1)))

    return pred_archive, sol_array, obj_array
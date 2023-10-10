###### Archive packages #####
import numpy as np
import gc

###### Import Custom Scripts ######
from xfoil.simulate_airfoils import xfoil
from xfoil.generate_airfoils import generate_parsec_coordinates
from gp.fit_gp_model import fit_gp_model
from gp.predict_objective import predict_objective
from utils.pprint_nd import pprint, pprint_fstring
from utils.utils import eval_xfoil_loop, maximize_obj_improvement, store_n_best_elites
from map_elites import map_elites

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
BATCH_SIZE = config.BATCH_SIZE
PRED_N_EVALS = config.PRED_N_EVALS
PRED_ELITE_REEVALS = config.PRED_ELITE_REEVALS
MAX_PRED_VERIFICATION = config.MAX_PRED_VERIFICATION


def prediction_verification_loop(pred_archive, obj_archive, pred_emitter, gp_model, sol_array, obj_array, extra_evals=0):

    print("\n\n ## Enter Prediction Verification Loop##")
    extra_evals = 0
    pred_n_evals = PRED_N_EVALS//(PRED_ELITE_REEVALS+1) # +1 because after the loop predictions with map_elites is called once more
    total_evals = PRED_N_EVALS

    while total_evals > pred_n_evals:

        total_evals -= pred_n_evals
        old_elites = np.array([(elite.solution, elite.index, elite.objective, elite.measures) for elite in obj_archive], dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])

        pred_archive, new_elite_archive = map_elites(pred_archive, pred_emitter, gp_model, pred_n_evals, predict_objective)                   # predict new elites
        max_obj_improvement_elites, new_elites = maximize_obj_improvement(new_elite_archive, old_elites)

        if MAX_PRED_VERIFICATION < extra_evals+len(max_obj_improvement_elites):
            Warning(f"MAX_PRED_VERIFICATION ({MAX_PRED_VERIFICATION}) exceeded. Exiting prediction verification loop.")
            break
        pred_archive, extra_evals, gp_model = prediction_verification(max_obj_improvement_elites, pred_archive, obj_archive, sol_array, obj_array, extra_evals, gp_model)  # verify predictions

    print(f"\n\nExtra evaluations (output): {extra_evals}\n\n")
    pred_archive, new_elite_archive = map_elites(pred_archive, pred_emitter, gp_model, pred_n_evals, predict_objective)

    new_elite_archive.clear()
    gc.collect()

    return pred_archive, extra_evals                                                                                                        # communicate extra evaluations


def prediction_verification(max_obj_improvement_elites, pred_archive, obj_archive, sol_array, obj_array, extra_evals, gp_model):
    """
    - Evaluates all new elites in the prediction archive.
    - Stores converged new elites if they present elite objectives.
    - Preserves obj_archive elites if predicted elites are not better.
    """

    print("n max obj improvment elites: " + str(len(max_obj_improvement_elites)))
    print(max_obj_improvement_elites)

    if len(max_obj_improvement_elites) == 0:
        return pred_archive, extra_evals, gp_model
    
    max_improvement_sol = np.vstack(max_obj_improvement_elites['solution'])
    max_improvement_bhv = np.vstack(max_obj_improvement_elites['behavior'])

    conv_sol, conv_obj, conv_bhv = eval_xfoil_loop(max_improvement_sol, max_improvement_bhv)     # evaluate in for loop to ensure BATCH_SIZE is not exceeded

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

    gp_model = fit_gp_model(sol_array, obj_array)                                                                                       # update GP model
    extra_evals += len(max_improvement_sol)                                                                                   # count extra evaluations

    return pred_archive, extra_evals, gp_model
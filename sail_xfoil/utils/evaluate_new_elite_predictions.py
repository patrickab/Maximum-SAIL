###### Archive packages #####
import numpy as np

###### Import Custom Scripts ######
from xfoil.simulate_airfoils import xfoil
from xfoil.generate_airfoils import generate_parsec_coordinates
from gp.fit_gp_model import fit_gp_model
from utils.pprint_nd import pprint, pprint_fstring
from utils.utils import eval_xfoil_loop

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
BATCH_SIZE = config.BATCH_SIZE


def evaluate_new_elite_predictions(new_elite_archive, pred_archive, obj_archive, sol_array, obj_array):
    """
    - Evaluates the new elites in the prediction archive.
    - Adds converged new elites if they present elite objectives
    - Preserves obj_archive elites if predicted elites are not better
    """

    new_elites = np.array(
        [(elite.solution, elite.index, elite.objective, elite.measures) for elite in new_elite_archive], 
        dtype=[('solution', object), ('index', int), ('prediction', float), ('behavior', object)])

    new_elite_sol = np.vstack(new_elites['solution'])
    new_elite_bhv = np.vstack(new_elites['behavior'])

    # evaluate in for loop to ensure BATCH_SIZE is not exceeded
    conv_sol, conv_obj, conv_bhv = eval_xfoil_loop(new_elite_sol, new_elite_bhv)

    obj_elite_sol = np.array([elite.solution for elite in obj_archive])
    obj_elite_obj = np.array([elite.objective for elite in obj_archive])
    obj_elite_bhv = np.array([elite.measures for elite in obj_archive])

    condidate_elite_sol = np.concatenate((obj_elite_sol, conv_sol))
    condidate_elite_obj = np.concatenate((obj_elite_obj, conv_obj))
    condidate_elite_bhv = np.concatenate((obj_elite_bhv, conv_bhv))

    pred_archive.clear()
    pred_archive.add(condidate_elite_sol, condidate_elite_obj, condidate_elite_bhv)
    # obj_archive.add(new_elite_solutions, new_elite_objectives, new_elite_measures) remove comment in future, for now keep it to improve comparability

    # store evaluations for GP model
    sol_array = np.vstack((sol_array, conv_sol))
    obj_array = np.vstack((obj_array, conv_obj.reshape(-1,1)))

    return pred_archive, sol_array, obj_array
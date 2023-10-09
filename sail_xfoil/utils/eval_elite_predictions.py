###### Archive packages #####
import numpy as np

###### Import Custom Scripts ######
from xfoil.simulate_airfoils import xfoil
from xfoil.generate_airfoils import generate_parsec_coordinates
from gp.fit_gp_model import fit_gp_model
from utils.pprint_nd import pprint, pprint_fstring

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
BATCH_SIZE = config.BATCH_SIZE


def evaluate_new_elite_predictions(new_elite_archive, pred_archive, obj_archive, gp_model, sol_array, obj_array):

    new_elites = np.array(
        [(elite.solution, elite.index, elite.objective, elite.measures) for elite in new_elite_archive], 
        dtype=[('solution', object), ('index', int), ('prediction', float), ('behavior', object)])
    
    valid_indices, surface_area_batch = generate_parsec_coordinates(new_elites['solution'])
    convergence_errors, success_indices, new_elite_objectives = xfoil(BATCH_SIZE)

    new_elite_solutions = new_elites['solution'][success_indices]
    new_elite_objectives = new_elite_objectives
    new_elite_measures = new_elites['behavior'][success_indices]

    pred_archive.add(new_elite_solutions, new_elite_objectives, new_elite_measures)
    # obj_archive.add(new_elite_solutions, new_elite_objectives, new_elite_measures) remove comment in future, for now keep it to improve comparability

    # store evaluations for GP model
    sol_array = np.vstack((sol_array, new_elite_solutions))
    obj_array = np.vstack((obj_array, new_elite_objectives.reshape(-1,1)))

    gp_model = fit_gp_model(sol_array, obj_array)

    return pred_archive, sol_array, obj_array, gp_model
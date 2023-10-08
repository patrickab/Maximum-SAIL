###### Archive packages #####
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from numpy import float64
import numpy as np
import subprocess
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
BATCH_SIZE = config.BATCH_SIZE
SOL_DIMENSION = config.SOL_DIMENSION
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SIGMA_EMITTER = config.SIGMA_EMITTER


def store_n_best_elites(archive, n, update_acq=True, gp_model=None):

    n_elites = sorted(archive, key=lambda x: x.objective, reverse=True)[:n]

    if update_acq:
        n_elite_acq = acq_ucb(np.array([elite.solution for elite in n_elites]), gp_model)
    else:
        n_elite_acq = [elite.objective for elite in n_elites]

    archive.clear()
    archive.add([elite.solution for elite in n_elites], n_elite_acq, [elite.measures for elite in n_elites])

    return archive


def sail_custom(acq_archive, obj_archive, gp_model, sol_array, obj_array):

    acq_emitter = define_acq_emitter(obj_archive, acq_archive, gp_model, seed=0)

    eval_budget = ACQ_N_OBJ_EVALS
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
        print("Obj Elites (before): " + str(obj_archive.stats.num_elites))
        status_vector, value_vector = obj_archive.add(new_elite_solutions[success_indices], new_elites_objectives, new_elite_measures[success_indices])
        print("Obj Elites (after): " + str(obj_archive.stats.num_elites))
        pprint_fstring(new_elite_acquisition, status_vector, new_elites_objectives)
        print("New Acq Elites: " + str(new_elite_archive.stats.num_elites))
        print("Acq Archive Size: " + str(acq_archive.stats.num_elites)) # print with only four decimals
        print("Acq QD (Custom): " +  str(int(acq_archive.stats.qd_score)))
        print("Airfoil Convergence Errors: " + str(convergence_errors))
        print("Remaining ACQ Precise Evals: " + str(eval_budget) + "\n\n")

        gp_model = fit_gp_model(sol_array, obj_array)

    return obj_archive, gp_model
    

def sail_vanilla(acq_archive, obj_archive, gp_model, sol_array, obj_array):

    acq_emitter = define_acq_emitter(obj_archive, acq_archive, gp_model, seed=0)

    eval_budget = ACQ_N_OBJ_EVALS
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
        convergence_errors, success_indices, obj_batch = xfoil(BATCH_SIZE)

        # store evaluations for GP model
        sol_array = np.vstack((sol_array, acq_elite_solutions[success_indices])) # dtype=float64
        obj_array = np.vstack((obj_array, obj_batch.reshape(-1,1))) # dtype=float64

        acq_batch = acq_elite_acquisitions[success_indices]
        print("Obj Elites (before): " + str(obj_archive.stats.num_elites))
        status_vector, value_vector = obj_archive.add(acq_elite_solutions[success_indices], obj_batch, acq_elite_measures[success_indices])
        print("Obj Elites (after): " + str(obj_archive.stats.num_elites))
        pprint_fstring(acq_batch, status_vector, obj_batch)
        print("Acq Archive Size: " + str(acq_archive.stats.num_elites))
        print("Acq QD (Vanilla): " +  str(int(acq_archive.stats.qd_score)))
        print("Airfoil Convergence Errors: " + str(convergence_errors))
        print("Remaining ACQ Precise Evals: " + str(eval_budget) + "\n\n")

        gp_model = fit_gp_model(sol_array, obj_array)

    return obj_archive, gp_model

def sail_random(acq_archive, obj_archive, gp_model, acq_emitter, sol_array, obj_array):
    eval_budget = ACQ_N_OBJ_EVALS
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


def define_acq_emitter(obj_archive, acq_archive, gp_model, seed):

    obj_elites = np.array([elite.solution for elite in obj_archive])
    obj_elites_acq = acq_ucb(obj_elites, gp_model)
    obj_elites_measures = np.array([elite.measures for elite in obj_archive])

    acq_archive.add(obj_elites, obj_elites_acq, obj_elites_measures)

    emitter = [
        GaussianEmitter(
        archive=acq_archive,
        sigma=SIGMA_EMITTER,
        bounds= np.array(SOL_VALUE_RANGE),
        batch_size=BATCH_SIZE,
        initial_solutions=obj_elites, # these solutions are never used, as the archive is never empty - however, specification is required for initializing the GaussianEmitter class
        seed=seed
    )]

    return emitter

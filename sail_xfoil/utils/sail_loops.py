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
from utils.utils import maximize_obj_improvement, store_n_best_elites


###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
ACQ_N_OBJ_EVALS = config.ACQ_N_OBJ_EVALS
ACQ_N_MAP_EVALS = config.ACQ_N_MAP_EVALS
BATCH_SIZE = config.BATCH_SIZE
SOL_DIMENSION = config.SOL_DIMENSION
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SIGMA_EMITTER = config.SIGMA_EMITTER


def sail_custom(acq_archive: GridArchive, obj_archive: GridArchive, gp_model, sol_array, obj_array, extra_evals):
    """
    Extra evaluations are given if eval_pred_flag is True (see sail.py)
        if not eval_pred_flag, extra_evals = 0
    """

    acq_emitter = define_acq_emitter(obj_archive, acq_archive, gp_model, seed=0)

    total_improvements = 0
    total_convergence_errors = 0
    mean_acq_improvement = 0
    eval_budget = ACQ_N_OBJ_EVALS + extra_evals
    while(eval_budget >= BATCH_SIZE):

        # store best elites from obj_archive in acq_archive & update acquisition values
        acq_archive = store_n_best_elites(acq_archive, obj_archive.stats.num_elites, update_acq=True, gp_model=gp_model, obj_archive=obj_archive)

        old_elites = np.array([(elite.solution, elite.index, elite.objective, elite.measures) for elite in acq_archive], dtype=[('solution', object), ('index', int), ('objective', float), ('behavior', object)])

        acq_archive, new_elite_archive = map_elites(acq_archive, acq_emitter, gp_model, ACQ_N_MAP_EVALS, acq_ucb)
        max_acq_improvement_elites, new_elites = maximize_obj_improvement(new_elite_archive, old_elites)

        # if there are not suffcient (BATCH_SIZE) acqisition improvements, re-enter acquisition loop
        condition_reached = False
        if len(max_acq_improvement_elites) >= BATCH_SIZE:
            max_acq_improvement_elites = max_acq_improvement_elites
            condition_reached = True

        if len(max_acq_improvement_elites)+len(new_elites) >= BATCH_SIZE and not condition_reached:
            max_acq_improvement_elites = np.concatenate((max_acq_improvement_elites, new_elites), axis=0)
            condition_reached = True

        else:
            iter = 0
            MAX_ITER = 10
            while len(max_acq_improvement_elites)+len(new_elites) < BATCH_SIZE:
                print("\n\n### Not enough Acq Improvements: Re-entering acquisition loop###\n\n")
                print("New Acq Elites found: " + str(len(max_acq_improvement_elites)))
                print("Archive Seed: " + str(acq_archive._seed))
                pprint(max_acq_improvement_elites)
                if iter >= MAX_ITER:
                    print("Max Iterations Reached: Exiting Acquisition Loop")
                    break
                iter += 1
                acq_archive, new_elite_archive = map_elites(acq_archive, acq_emitter, gp_model, ACQ_N_MAP_EVALS, acq_ucb, new_elite_archive=new_elite_archive)
                max_acq_improvement_elites_2, new_elites_2 = maximize_obj_improvement(new_elite_archive, old_elites)
                max_acq_improvement_elites = np.concatenate((max_acq_improvement_elites, max_acq_improvement_elites_2, new_elites_2), axis=0) # code doesnt ensure that behaviorally different elites are selected - same bin can be selected multiple times

        # select BATCH_SIZE acqisition elites, sorted by acquisition improvement
        max_acq_improvement_elites = max_acq_improvement_elites[np.argsort(max_acq_improvement_elites['objective_improvement'])]
        max_acq_improvement_elites = np.flip(max_acq_improvement_elites)
        max_acq_improvement_batch = max_acq_improvement_elites[:BATCH_SIZE]

        new_elite_solutions = np.vstack(max_acq_improvement_batch['solution'])
        acquisition_improvement = np.vstack(max_acq_improvement_batch['objective_improvement'])
        new_elite_measures = np.vstack(max_acq_improvement_batch['behavior'])

        # acq_archive only contains valid solutions (= non-intersecting polynomials), therefore no need to check for validity using valid_indices
        valid_indices, surface_area_batch = generate_parsec_coordinates(new_elite_solutions)
        convergence_errors, success_indices, new_elites_objectives = xfoil(BATCH_SIZE)

        # if acq_elite doesnt converge, remove from acq_archive
        new_elite_solutions = new_elite_solutions[success_indices]
        new_elite_acq = acquisition_improvement[success_indices]
        new_elite_measures = new_elite_measures[success_indices]
        # add acq_elites

        # store evaluations for GP model
        sol_array = np.vstack((sol_array, new_elite_solutions)) # dtype=float64
        obj_array = np.vstack((obj_array, new_elites_objectives.reshape(-1,1))) # dtype=float64

        eval_budget -= BATCH_SIZE

        acquisition_improvement = acquisition_improvement[success_indices]
        status_vector, value_vector = obj_archive.add(new_elite_solutions, new_elites_objectives, new_elite_measures)

        total_improvements += np.sum(status_vector > 0)
        total_convergence_errors += np.sum(convergence_errors)
        mean_acq_improvement += np.mean(acquisition_improvement)/(BATCH_SIZE-convergence_errors) # referring only to converged & evaluated acquisition elites
        mean_obj_improvement = np.mean(new_elites_objectives)/(BATCH_SIZE-convergence_errors)
        percentage_improvements = (total_improvements/(ACQ_N_OBJ_EVALS-eval_budget))*100
        percentage_convergence_errors = (total_convergence_errors/(ACQ_N_OBJ_EVALS-eval_budget))*100

        print("Total Improvements: " + str(total_improvements))
        print("Total Convergence Errors: " + str(total_convergence_errors))
        print("Percentage Improvements: {:.1f}".format(percentage_improvements) + "%")
        print("Percentage Convergence Errors: {:.1f}".format(percentage_convergence_errors) + "%")
        print("Mean Acq Improvement: {:.1f}".format(mean_acq_improvement))
        print("Mean Obj Improvement: {:.1f}".format(mean_obj_improvement))
        print("Status Vector: " + str(status_vector))
        pprint_fstring(acquisition_improvement, new_elites_objectives)
        print("New Acq Elites: " + str(new_elite_archive.stats.num_elites))
        print("Acq Archive Size: " + str(acq_archive.stats.num_elites))
        print("Obj Archive Size: " + str(obj_archive.stats.num_elites))
        print("Airfoil Convergence Errors: " + str(convergence_errors))
        print("Remaining ACQ Precise Evals: " + str(eval_budget) + "\n\n")

        gp_model = fit_gp_model(sol_array, obj_array)

    return obj_archive, gp_model
    

def sail_vanilla(acq_archive, obj_archive, gp_model, sol_array, obj_array, extra_evals):
    """
    Extra evaluations are given if eval_pred_flag is True (see sail.py)
        if not eval_pred_flag, extra_evals = 0
    """

    print(f"\n\nExtra evaluations (input): {extra_evals}\n\n")


    acq_emitter = define_acq_emitter(obj_archive, acq_archive, gp_model, seed=0)

    total_improvements = 0
    total_convergence_errors = 0
    mean_acq_improvement = 0
    eval_budget = ACQ_N_OBJ_EVALS + extra_evals
    while(eval_budget >= BATCH_SIZE):
        eval_budget -= BATCH_SIZE

        # store best elites from obj_archive in acq_archive & update acquisition values
        acq_archive = store_n_best_elites(acq_archive, obj_archive.stats.num_elites, update_acq=True, gp_model=gp_model, obj_archive=obj_archive)
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
        status_vector, value_vector = obj_archive.add(acq_elite_solutions[success_indices], obj_batch, acq_elite_measures[success_indices])
        total_improvements += np.sum(status_vector > 0)
        total_convergence_errors += np.sum(convergence_errors)
        mean_obj_improvement = np.mean(obj_batch)/(BATCH_SIZE-convergence_errors)
        percentage_improvements = (total_improvements/(ACQ_N_OBJ_EVALS-eval_budget))*100
        percentage_convergence_errors = (total_convergence_errors/(ACQ_N_OBJ_EVALS-eval_budget))*100

        print("Total Improvements: " + str(total_improvements))
        print("Total Convergence Errors: " + str(total_convergence_errors))
        print("Percentage Improvements: {:.1f}".format(percentage_improvements) + "%")
        print("Percentage Convergence Errors: {:.1f}".format(percentage_convergence_errors) + "%")
        print("Mean Obj Improvement: {:.1f}".format(mean_obj_improvement))
        print("Status Vector: " + str(status_vector))
        pprint_fstring(acq_batch, obj_batch)
        print("Acq Archive Size: " + str(acq_archive.stats.num_elites))
        print("Obj Archive Size: " + str(obj_archive.stats.num_elites))
        print("Airfoil Convergence Errors: " + str(convergence_errors))
        print("Remaining ACQ Precise Evals: " + str(eval_budget) + "\n\n")

        gp_model = fit_gp_model(sol_array, obj_array)

    return obj_archive, gp_model

def sail_random(acq_archive, obj_archive, gp_model, sol_array, obj_array, extra_evals):
    """
    Extra evaluations are given if eval_pred_flag is True (see sail.py)
        if not eval_pred_flag, extra_evals = 0
    """


    eval_budget = ACQ_N_OBJ_EVALS + extra_evals
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
    """Reduces Overhead"""

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

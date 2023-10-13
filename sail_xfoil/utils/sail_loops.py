def run_vanilla_sail(acq_archive, obj_archive, gp_model, sol_array, obj_array, extra_evals, initial_seed, benchmark_domain):
    """
    Extra evaluations are given if eval_pred_flag is True (see sail.py)
        if not eval_pred_flag, extra_evals = 0
    """

    print(f"\n\nExtra evaluations (input): {extra_evals}\n\n")


    acq_emitter = update_emitter(obj_archive, acq_archive, gp_model, seed=0)

    total_improvements = 0
    total_convergence_errors = 0
    mean_acq_improvement = 0
    eval_budget = ACQ_N_OBJ_EVALS + extra_evals
    while(eval_budget >= BATCH_SIZE):
        eval_budget -= BATCH_SIZE

        # store best elites from obj_archive in acq_archive & update acquisition values

        #acq_archive = store_n_best_elites(acq_archive, obj_archive.stats.num_elites, update_acq=True, gp_model=gp_model, obj_archive=obj_archive)
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

        anytime_archive_visualizer(archive=obj_archive, benchmark_domain=benchmark_domain, initial_seed=initial_seed, iteration=(ACQ_N_OBJ_EVALS+extra_evals-eval_budget)//BATCH_SIZE)

        gp_model = fit_gp_model(sol_array, obj_array)

    return obj_archive, gp_model


def run_random_sail(acq_archive, obj_archive, gp_model, sol_array, obj_array, extra_evals, initial_seed, benchmark_domain):
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
        anytime_archive_visualizer(archive=obj_archive, benchmark_domain=benchmark_domain, initial_seed=initial_seed, iteration=(ACQ_N_OBJ_EVALS+extra_evals-eval_budget)//BATCH_SIZE)

        print("\n\nAirfoil Convergence Errors: " + str(convergence_errors))
        print("Remaining Random Search Obj Evals: " + str(eval_budget) + "\n\n")

    return obj_archive, gp_model

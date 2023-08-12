# elite_status_vector = archive.add(acq_elites, obj_evals, bhv_evals)
        # elite_status_vector == 0  ->  acq_elite was not added
        # elite_status_vector == 1  ->  acq_elite was added
        # elite_status_vector == 2  ->  acq_elite discovered new cell

def select_new_elites(candidate_elites, obj_evals, elite_status_vector):

    x_new_elites = []
    obj_new_elites = []

    for candidate_elite, obj_eval ,elite_status_value in candidate_elites, obj_evals, elite_status_vector:
        if elite_status_value > 0:
            x_new_elites.append(candidate_elite)
            obj_new_elites.append(obj_eval)

    return x_new_elites, obj_new_elites

#def count_new_elites(elite_status_vector):
    
    
import numpy
from numpy import float64

def example_objective_function(samples):
    
    if samples is None:
        return

    obj_evals = []

    for sample in samples:
        obj_evals.append(numpy.random.normal(0,1))
    
    obj_evals = numpy.array(obj_evals)

    return obj_evals

def example_behavior_function(samples):

    if samples is None:
        return

    bhv_evals = [numpy.array([], dtype=float), numpy.array([], dtype=float64)]

    for sample in samples:
        # nothing meaningful being done, just calculates "random" behavioral values within value range from given sample input    
        dim1_bhv_eval = 0
        dim2_bhv_eval = 0

        for dim in sample:
            dim1_bhv_eval += (dim*3.1415 + numpy.random.normal(20, 9)) % 100 # generates value between 0 and 100
            dim2_bhv_eval += (dim*2.7182 + numpy.random.normal(10, 2)) % 20 # generates value between 0 and 20
        
        bhv_evals[0] = numpy.append(bhv_evals[0], dim1_bhv_eval)
        bhv_evals[1] = numpy.append(bhv_evals[1], dim2_bhv_eval)

    bhv_evals = numpy.array(bhv_evals, dtype=float)

    return bhv_evals.T

def example_variation_function(samples):
    
    if samples is None:
        return

    print(samples)

    for sample in samples:
        i=0
        for dim in sample:
            if i%2 == 0:
                dim += 0.1
            else:
                dim -= 0.1
            i += 1

    print(samples)

    return samples
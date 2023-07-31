MEAN_WEIGHTING        = 1
UNCERTAINTY_WEIGHTING = 5

def acquisition_ucb(drag):
    """
    python translation of velo_AcquisitionFunc()

    Note: 
        simplified input values 
            here:     2x1 matrices
            original: 2xN matrices
        renamed variables
            fitness -> acq_fitness
            predValue -> acq_behavior        

    Inputs: 
        drag : (mean, variance)   ->  (Luftwiderstand)
        lift : (mean, variance)   ->  (Auftrieb)
    
    Outputs:
        acq_fitness : float
            Fitness value (lower drag is better)

        pred_value : 
            Predicted drag force (mean and variance)
    """

    acq_fitness  = (drag[0] * MEAN_WEIGHTING) - (drag[1] * UNCERTAINTY_WEIGHTING)
    acq_behavior = [drag[0], drag[1]]

    return acq_fitness, acq_behavior
import random

def acq_normal_distribution(genome):
    """No meaningful calculation, just calculates "random" acq(genome)~N(0,1)"""

    acq_fitness = random.normalvariate(0, 1)

    return acq_fitness
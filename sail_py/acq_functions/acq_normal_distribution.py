import random
import torch
from torch import float64

def acq_normal_distribution(genomes, gp_model):
    """No meaningful calculation, just calculates "random" acq(genome)~N(0,1)"""

    acq_fitness = torch.tensor([random.normalvariate(0, 1) for genome in genomes], dtype=float64)

    return acq_fitness
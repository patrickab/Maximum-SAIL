### Packages ###
import os
import torch
import numpy
from torch import float64
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

### Custom Scripts ###
from utils.pprint import pprint

### Global Ressources ###
from config import Config
config = Config(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SOL_DIMENSION = config.SOL_DIMENSION
PARALLEL_BATCH_SIZE = config.PARALLEL_BATCH_SIZE

def acq_ucb(genomes, gp_model):

    print("\nInitialize acq_ucb() [...]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    UCB = UpperConfidenceBound(gp_model, beta=0.2)

    genomes = torch.tensor(genomes, dtype=float64, device=device)           # Shape: PARALLEL_BATCH_SIZE x SOL_DIMENSION
    transformed_genomes = genomes.unsqueeze(1)                              # Shape: PARALLEL_BATCH_SIZE x 1 x SOL_DIMENSION

    ucb_tensor = UCB(transformed_genomes)

    print("[...] Terminate acq_ucb()\n")

    return ucb_tensor.detach().numpy()
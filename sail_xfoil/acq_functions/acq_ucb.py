from torch import float64, tensor
from botorch.acquisition import UpperConfidenceBound

from config.config import Config
config = Config('config/config.ini')
SIGMA_UCB = config.SIGMA_UCB

def acq_ucb(self, genomes, sigma_mutants=None, niche_restricted_update=None):

    genomes = tensor(genomes, dtype=float64)
    transformed_genomes = genomes.unsqueeze(1)

    UCB = UpperConfidenceBound(self.gp_model, beta=SIGMA_UCB)
    ucb_tensor = UCB(transformed_genomes)
    ucb_ndarray = ucb_tensor.detach().numpy()

    return ucb_ndarray
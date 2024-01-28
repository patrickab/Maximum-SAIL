### Packages ###
from torch import float64, cuda, device, tensor
from botorch.acquisition import UpperConfidenceBound

### Custom Scripts ###w
from utils.pprint_nd import pprint

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
SIGMA_UCB = config.SIGMA_UCB


def acq_ucb(self, genomes, sigma_mutants=None):

    genomes = tensor(genomes, dtype=float64)          # Shape: PARALLEL_BATCH_SIZE x SOL_DIMENSION
    transformed_genomes = genomes.unsqueeze(1)        # Shape: PARALLEL_BATCH_SIZE x 1 x SOL_DIMENSION

    UCB = UpperConfidenceBound(self.gp_model, beta=SIGMA_UCB)
    ucb_tensor = UCB(transformed_genomes)
    ucb_ndarray = ucb_tensor.detach().numpy()

    return ucb_ndarray
### Packages ###
from torch import float64, cuda, device, tensor
from botorch.acquisition import UpperConfidenceBound

### Custom Scripts ###w
from utils.pprint_nd import pprint

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
SIGMA_UCB = config.SIGMA_UCB


def acq_ucb(genomes, gp_model):

    dev = device("cuda" if cuda.is_available() else "cpu")

    UCB = UpperConfidenceBound(gp_model, beta=SIGMA_UCB)

    genomes = tensor(genomes, dtype=float64, device=dev)           # Shape: PARALLEL_BATCH_SIZE x SOL_DIMENSION
    transformed_genomes = genomes.unsqueeze(1)                     # Shape: PARALLEL_BATCH_SIZE x 1 x SOL_DIMENSION

    ucb_tensor = UCB(transformed_genomes)
    ucb_ndarray = ucb_tensor.detach().numpy()

    return ucb_ndarray
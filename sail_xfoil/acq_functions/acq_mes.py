### Packages ###
from torch import float64, cuda, device, tensor
from botorch.acquisition import qMaxValueEntropy

### Custom Scripts ###w
from utils.pprint_nd import pprint

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')
SIGMA_UCB = config.SIGMA_UCB


def acq_mes(genomes, gp_model):

    dev = device("cuda" if cuda.is_available() else "cpu")

    genomes = tensor(genomes, dtype=float64, device=dev)              # Shape: PARALLEL_BATCH_SIZE x SOL_DIMENSION
    transformed_genomes = genomes.unsqueeze(1)                        # Shape: PARALLEL_BATCH_SIZE x 1 x SOL_DIMENSION

    MES = qMaxValueEntropy(gp_model, genomes)
    mes_tensor = MES(transformed_genomes) 
    mes_ndarray = mes_tensor.detach().numpy()

    return mes_ndarray
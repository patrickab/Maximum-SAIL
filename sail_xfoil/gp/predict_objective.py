import torch
import gpytorch
import numpy
from torch import float64

from acq_functions.acq_ucb import acq_ucb
from xfoil.generate_airfoils import generate_parsec_coordinates
from utils.pprint_nd import pprint

# xfoil parameters
from config.config import Config
config = Config('config/config.ini')
ALFA = config.ALFA
MACH = config.MACH
REYNOLDS = config.REYNOLDS

def predict_objective(self, genomes):

    self.gp_model.eval()
    genomes_tensor = torch.tensor(genomes, dtype=float64)           # Shape: PARALLEL_BATCH_SIZE x SOL_DIMENSION
    transformed_genomes = genomes_tensor.unsqueeze(1)               # Shape: PARALLEL_BATCH_SIZE x 1 x SOL_DIMENSION

    # Get the predictive posterior distribution
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        posterior = self.gp_model.posterior(transformed_genomes)

    posterior_mean_predictions = posterior.mean.numpy()
    posterior_mean_predictions = numpy.array([prediction[0] for prediction_array in posterior_mean_predictions for prediction in prediction_array]).T

    return posterior_mean_predictions
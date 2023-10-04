import torch
import gpytorch
import numpy
from torch import float64

from acq_functions.acq_ucb import acq_ucb
from xfoil.generate_airfoils import generate_parsec_coordinates
from utils.pprint_nd import pprint, pprint_nd

# xfoil parameters
from config.config import Config
config = Config('config/config.ini')
ALFA = config.ALFA
MACH = config.MACH
REYNOLDS = config.REYNOLDS

def predict_objective(genomes, gp_model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gp_model.eval()
    genomes_tensor = torch.tensor(genomes, dtype=float64, device=device)           # Shape: PARALLEL_BATCH_SIZE x SOL_DIMENSION
    transformed_genomes = genomes_tensor.unsqueeze(1)                              # Shape: PARALLEL_BATCH_SIZE x 1 x SOL_DIMENSION

    # Get the predictive posterior distribution
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        posterior = gp_model.posterior(transformed_genomes)

    posterior_mean_predictions = posterior.mean.numpy()

    posterior_mean_predictions = numpy.array([prediction[0] for prediction_array in posterior_mean_predictions for prediction in prediction_array]).T

    pred_obj = posterior_mean_predictions #[success_indices]

    #pred_error = true_obj-pred_obj    
    # Stack the arrays horizontally
    #stacked_arrays = numpy.column_stack((true_obj, pred_obj, pred_error))

    # Define the format strings
    #format_string_names = f"True Obj:\tPred Obj:\tPred Error:"
    #format_string = "\n".join(["\t".join(map(str, row)) for row in stacked_arrays])
    #format_string = "\n".join(["\t\t".join([f"{value:.4f}" for value in row]) for row in stacked_arrays])

    return posterior_mean_predictions
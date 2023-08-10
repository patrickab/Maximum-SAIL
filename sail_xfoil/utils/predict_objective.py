import torch
import gpytorch
import numpy
from torch import float64

from acq_functions.acq_ucb import acq_ucb
from utils.simulate_robotarm import simulate_obj
from utils.pprint import pprint

def predict_objective(genomes, gp_model):

    print("\nInitialize predict_objective() [...]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gp_model.eval()
    genomes = torch.tensor(genomes, dtype=float64, device=device)           # Shape: PARALLEL_BATCH_SIZE x SOL_DIMENSION
    transformed_genomes = genomes.unsqueeze(1)                              # Shape: PARALLEL_BATCH_SIZE x 1 x SOL_DIMENSION

    # Get the predictive posterior distribution
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        posterior = gp_model.posterior(transformed_genomes)

    mean_predictions = posterior.mean.numpy()

    mean_predictions = numpy.array([prediction[0] for prediction_array in mean_predictions for prediction in prediction_array]).T

    true_obj = simulate_obj(genomes)
    pred_obj = mean_predictions
    pred_error = true_obj-pred_obj

    # Stack the arrays horizontally
    stacked_arrays = numpy.column_stack((true_obj, pred_obj, pred_error))

    # Define the format strings
    format_string_names = f"True Obj:\tPred Obj:\tPred Error:"
    format_string = "\n".join(["\t".join(map(str, row)) for row in stacked_arrays])
    format_string = "\n".join(["\t\t".join([f"{value:.4f}" for value in row]) for row in stacked_arrays])

    print("\nRobot Arm Results:\n")
    print(format_string_names)
    print(format_string)

    print("\n\n[...] Terminate predict_objective()\n")

    return mean_predictions
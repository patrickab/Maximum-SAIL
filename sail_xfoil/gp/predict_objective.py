import numpy
import torch
import gpytorch
from torch import float64

def predict_objective(self, genomes, sigma_mutants=None, niche_restricted_update=None):

    self.gp_model.eval()
    genomes_tensor = torch.tensor(genomes, dtype=float64)
    transformed_genomes = genomes_tensor.unsqueeze(1)

    # Get the predictive posterior distribution
    with gpytorch.settings.fast_pred_var(), torch.no_grad():
        posterior = self.gp_model.posterior(transformed_genomes)

    posterior_mean_predictions = posterior.mean.numpy()
    posterior_mean_predictions = numpy.array([prediction[0] for prediction_array in posterior_mean_predictions for prediction in prediction_array]).T

    return posterior_mean_predictions


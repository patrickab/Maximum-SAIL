import botorch
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# 'gp_observations = (new_solutions, new_obj_evals)'
#   'with new_solutions = scheduler.ask()'



def create_gp_model(X, Y):
    # X: torch.Tensor of shape (num_samples, num_features)
    # Y: torch.Tensor of shape (num_samples,)
    gp_model = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    return gp_model, mll


def update_gp_model(gp_model, mll, new_solutions, new_obj_evals):
    # new_solutions: torch.Tensor of shape (num_new_samples, num_features)
    # new_obj_evals: torch.Tensor of shape (num_new_samples,)

    # Append the new observations to the existing data
    X = torch.cat([gp_model.train_inputs[0], new_solutions])
    Y = torch.cat([gp_model.train_targets, new_obj_evals])

    # Update the GP model with new data
    gp_model = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

    # Re-fit the GP model to the updated data   	
    fit_gpytorch_model(mll)

    return gp_model, mll


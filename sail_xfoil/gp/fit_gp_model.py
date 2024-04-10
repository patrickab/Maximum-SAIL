import os
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize, Normalize

from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SOL_DIMENSION = config.SOL_DIMENSION

def fit_gp_model(x_sol, y_obj):
    n_data_points = len(x_sol)

    print("\nfit_gp_model()...")
    print(f"n_data_points: {n_data_points}\n")

    dtype = torch.float64

    # Convert variables to tensor
    x_tensor = torch.tensor(x_sol, dtype=dtype)
    y_tensor = torch.tensor(y_obj, dtype=dtype)

    lower_bounds = torch.tensor([lower for lower, _ in SOL_VALUE_RANGE])
    upper_bounds = torch.tensor([upper for _, upper in SOL_VALUE_RANGE])
    bounds = torch.stack((lower_bounds, upper_bounds))
    
    input_transform   = Normalize(d=SOL_DIMENSION, bounds=bounds)
    outcome_transform = Standardize(m=1)

    gp_model = SingleTaskGP(train_X=x_tensor, train_Y=y_tensor, input_transform=input_transform, outcome_transform=outcome_transform)

    # Define marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood=gp_model.likelihood, model=gp_model)

    fit_gpytorch_model(mll)

    return gp_model
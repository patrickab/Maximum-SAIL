import torch
import os
from botorch.models.transforms import Standardize, Normalize
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood


### Custom Scripts ###
from utils.pprint import pprint

### Global Ressources ###
from config import Config
config = Config(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SOL_DIMENSION = config.SOL_DIMENSION

def fit_gp_model(x_solutions, y_obj_evals):

    print("\nInitialize fit_gp_model [...]")

    # Use a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    x_tensor = torch.tensor(x_solutions, device=device, dtype=dtype)
    y_tensor = torch.tensor(y_obj_evals, device=device, dtype=dtype)

    # Define Bounds for Normalize()
    lower_bounds = torch.tensor([lower for lower, _ in SOL_VALUE_RANGE])
    upper_bounds = torch.tensor([upper for _, upper in SOL_VALUE_RANGE])
    bounds = torch.stack((lower_bounds, upper_bounds))
    # Define Normalize() for GP
    input_transform = Normalize(d=SOL_DIMENSION, bounds=bounds) #torch.stack([torch.zeros(SOL_DIMENSION), input_ranges]))

    gp_model = SingleTaskGP(train_X=x_tensor, train_Y=y_tensor, input_transform=input_transform, outcome_transform=Standardize(m=1))

    # Define marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood=gp_model.likelihood, model=gp_model)
    mll.to(x_tensor)

    fit_gpytorch_model(mll)

    print("[...] Terminate fit_gp_model\n")
    return gp_model
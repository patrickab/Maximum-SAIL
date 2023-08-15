import torch
import os
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms import Standardize, Normalize
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood


### Custom Scripts ###
from utils.pprint import pprint

### Global Ressources ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE
SOL_DIMENSION = config.SOL_DIMENSION

def fit_gp_model(x_solutions, y_obj_evals):

    print("\nfit_gp_model()...\n\n")

    # Use a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    # Convert variables to tensor
    x_tensor = torch.tensor(x_solutions, device=device, dtype=dtype)
    y_tensor = torch.tensor(y_obj_evals, device=device, dtype=dtype)
    
    lower_bounds = torch.tensor([lower for lower, _ in SOL_VALUE_RANGE])
    upper_bounds = torch.tensor([upper for _, upper in SOL_VALUE_RANGE])
    bounds = torch.stack((lower_bounds, upper_bounds))

    input_transform = Normalize(d=SOL_DIMENSION, bounds=bounds)

    gp_model = SingleTaskGP(train_X=x_tensor, train_Y=y_tensor, input_transform=input_transform, outcome_transform=Standardize(m=1))

    # RISES ERROR:    gp_model = SingleTaskGP(train_X=x_tensor, train_Y=y_tensor, input_transform=Normalize(x_tensor.shape[-1], bounds=sol_value_range_tensor), outcome_transform=Standardize(m=1))
    # ERROR:    botorch.exceptions.errors.BotorchTensorDimensionError: Dimensions of provided `bounds` are incompatible with transform_dimension = 0!

    # Define marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood=gp_model.likelihood, model=gp_model)
    mll.to(x_tensor)

    fit_gpytorch_model(mll)

    return gp_model
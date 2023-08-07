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

def fit_gp_model(x_solutions, y_obj_evals):

    print("\nInitialize fit_gp_model [...]")

    # Use a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64


    pprint(x_solutions)
    print()
    pprint(y_obj_evals)
    # Convert variables to tensor if necessary
    x_tensor = torch.tensor(x_solutions, device=device, dtype=dtype)
    y_tensor = torch.tensor(y_obj_evals, device=device, dtype=dtype)

    sol_value_range_tensor = torch.tensor([list(t) for t in SOL_VALUE_RANGE], dtype=torch.float64)

    # Initialize model
    gp_model = SingleTaskGP(train_X=x_tensor, train_Y=y_tensor)

    # RISES ERROR:    gp_model = SingleTaskGP(train_X=x_tensor, train_Y=y_tensor, input_transform=Normalize(x_tensor.shape[-1], bounds=sol_value_range_tensor), outcome_transform=Standardize(m=1))
    # ERROR:    botorch.exceptions.errors.BotorchTensorDimensionError: Dimensions of provided `bounds` are incompatible with transform_dimension = 0!

    # Define marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood=gp_model.likelihood, model=gp_model)
    mll.to(x_tensor)

    fit_gpytorch_model(mll)

    print("[...] Terminate fit_gp_model\n")
    return gp_model
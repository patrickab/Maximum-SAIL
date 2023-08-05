import torch
from botorch.models.transforms import Standardize, Normalize
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from ..config import Config
config = Config('../config.ini')
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE

def fit_gp_model(obj_eval_archive):

    # Use a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    x, y = obj_eval_archive

    # Convert variables to tensor if necessary
    x_tensor = torch.tensor(x, device=device, dtype=dtype)
    y_tensor = torch.tensor(y, device=device, dtype=dtype)

    # Initialize model
    gp_model = SingleTaskGP(train_X=x_tensor, train_Y=y_tensor, input_transform=Normalize, output_transform=Standardize)
    
    # Define marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood=gp_model.likelihood, model=gp_model)
    mll.to(x)

    fit_gpytorch_model(mll)

    return gp_model


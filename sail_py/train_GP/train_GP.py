import torch
from botorch.models.transforms import Standardize, Normalize
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood

from ..config import Config
config = Config('../config.ini')
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE

def init_gp_model(init_solutions, obj_evals):

    # Use a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    # Convert variables to tensor if necessary
    init_solutions = torch.tensor(init_solutions, device=device, dtype=dtype)
    obj_evals = torch.tensor(obj_evals, device=device, dtype=dtype)

    # Initialize model
    gp_model = SingleTaskGP(train_X=init_solutions, train_Y=obj_evals, input_transform=Normalize) # "output_transform = " torch tensor containing bounds
    
    # Define marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood=gp_model.likelihood, model=gp_model)

    fit_gpytorch_model(mll)

    return gp_model


def update_gp_model(obj_eval_archive, gp_model):

    # Use a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    x, y = obj_eval_archive

    # Convert variables to tensor if necessary
    x_tensor = torch.tensor(x, device=device, dtype=dtype)
    y_tensor = torch.tensor(y, device=device, dtype=dtype)

    # Update the GP model with new data
        # How can I fit GP on existing gp_model?
    gp_model = SingleTaskGP(train_X=x, train_Y=y)
    mll = ExactMarginalLogLikelihood(likelihood=gp_model.likelihood, model=gp_model)
    mll.to(x) # ???

    # Re-fit the GP model to the updated data   	
    fit_gpytorch_model(mll)

    return gp_model


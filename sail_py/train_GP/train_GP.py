import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan

# 'gp_observations = (new_solutions, new_obj_evals)'
#   'with new_solutions = scheduler.ask()'


def init_gp_model(init_solutions, obj_evals):

    # Use a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float

    # Convert variables to tensor if necessary
    init_solutions = torch.tensor(init_solutions, device=device, dtype=dtype)
    obj_evals = torch.tensor(obj_evals, device=device, dtype=dtype)

    # Initialize model
    gp_model = SingleTaskGP(train_X=init_solutions, train_Y=obj_evals)
    gp_model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5)) # ???
    
    # Define marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood=gp_model.likelihood, model=gp_model)
    mll.to(init_solutions) # ???

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


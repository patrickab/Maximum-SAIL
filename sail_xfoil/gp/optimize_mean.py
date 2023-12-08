import os
import numpy as np
import torch
from botorch.acquisition import PosteriorMean
from botorch.optim import optimize_acqf

### Global Parameters ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE

def maximize_mean(gp_model):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acq_func = PosteriorMean(gp_model)
    new_x, max_mean = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.tensor(SOL_VALUE_RANGE, dtype=torch.float64, device=device).T,
        q=1,
        num_restarts=10,
        raw_samples=1024,
    )

    max_mean = max_mean.detach().numpy()
    new_x = new_x.detach().numpy()

    return new_x, max_mean

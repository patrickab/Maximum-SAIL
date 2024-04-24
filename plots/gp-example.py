import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch.models.transforms import Standardize, Normalize
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
import sobol_seq


# Define the objective function
def objective_function(x):
    return 0.5 * x**2 * np.sin(x)

# Define the search space
lower_bound = -10.0
upper_bound = 10.0
space = [(lower_bound, upper_bound)]  # Bounds for x

# Generate random design points
n_design_points = 15
X_design = np.random.uniform(low=-10, high=10, size=(n_design_points, 1))

# X_design = sobol_seq.i4_sobol_generate(1, n_design_points)
# X_design = lower_bound + (upper_bound - lower_bound) * X_design

# Evaluate objective function at design points
Y_design = [objective_function(x) for x in X_design]

# Plot the design points
x = np.linspace(-10, 10, 1000)
y = objective_function(x)
plt.plot(x, y, 'r-', label='Objective Function')
plt.scatter(X_design, Y_design, color='blue', marker='o', label='Design Points')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Objective Function Evaluated on Uniform Random Sample (n=15)')
plt.ylim(-35, 35)
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('bo-example-objective.png')

plt.close()

def fit_gp_model(x_sol, y_obj):
    x_tensor = torch.tensor(x_sol, dtype=torch.float64)
    y_tensor = torch.tensor(y_obj, dtype=torch.float64)

    l_bound = torch.tensor([lower_bound], dtype=torch.float64)
    u_bound = torch.tensor([upper_bound], dtype=torch.float64)
    bounds = torch.stack([l_bound, u_bound])

    input_transform = Normalize(d=1, bounds=bounds)

    gp_model = SingleTaskGP(x_tensor, y_tensor,
        input_transform=input_transform,
        outcome_transform=Standardize(m=1))

    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

    fit_gpytorch_model(mll)

    return gp_model

# Fit a gaussian process model using botorch to the design points, plot the mean and the 95% confidence interval
gp_model = fit_gp_model(X_design, Y_design)
# Plot the model and its credible intervals
x = np.linspace(-10, 10, 1000)
x_tensor = torch.tensor(x, dtype=torch.float64).view(-1, 1)
with torch.no_grad():
    f_mean = gp_model.posterior(x_tensor).mean
    f_std = gp_model.posterior(x_tensor).stddev

f_mean = gp_model.posterior(x_tensor).mean.squeeze().detach().numpy()
f_std = gp_model.posterior(x_tensor).stddev.squeeze().detach().numpy()

plt.plot(x, f_mean, 'r-', label='GP Mean')
plt.fill_between(x, f_mean - 1.96 * f_std, f_mean + 1.96 * f_std, alpha=0.2, label='95% Credible Interval')
plt.scatter(X_design, Y_design, color='blue', marker='o', label='Design Points')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('GP Model and 95% Credible Interval')
plt.ylim(-35, 35)
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('bo-example-gp.png')

plt.close()

import subprocess
subprocess.call(["convert", "+append", "bo-example-objective.png", "bo-example-gp.png", "bo-example.png"]) 

from botorch.acquisition import UpperConfidenceBound
def acq_ucb(gp_model):
    x = np.linspace(-10, 10, 1000)
    x_tensor = torch.tensor(x, dtype=torch.float64).unsqueeze(1).unsqueeze(2)
    UCB = UpperConfidenceBound(gp_model, beta=2)
    ucb_tensor = UCB(x_tensor)
    return ucb_tensor.detach().numpy()

# Plot the UCB acquisition function
ucb = acq_ucb(gp_model)
plt.plot(x, ucb, 'r-', label='UCB')
plt.xlabel('x')
plt.ylabel('UCB(x)')
plt.title('UCB Acquisition Function')
plt.ylim(-35, 35)
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('bo-example-ucb.png')

subprocess.call(["convert", "+append", "bo-example-gp.png", "bo-example-ucb.png", "bo-example-gp-ucb.png"])

plt.close()

from botorch.acquisition import qLowerBoundMaxValueEntropy
def simple_mes(gp_model):
    x = np.linspace(-10, 10, 1000)
    x_tensor = torch.tensor(x, dtype=torch.float64)
    transformed_x = x_tensor.unsqueeze(1)
    acq_solution_tensor = torch.tensor(np.zeros(len(x)), dtype=torch.float64)
    acq_entropy_tensor = torch.tensor(np.zeros((len(x), 1)), dtype=torch.float64)
    MES = qLowerBoundMaxValueEntropy(model=gp_model, candidate_set=transformed_x, num_mv_samples=200)
    for i in range(x.shape[0]):
        acq_entropy = MES(transformed_x[i].unsqueeze(0))  # Reshape to have 2 dimensions
        acq_entropy_tensor[i] = acq_entropy
        acq_solution_tensor[i] = x_tensor[i]
    mes_ndarray = acq_entropy_tensor.detach().numpy()
    return np.hstack(mes_ndarray)

subprocess.call(["convert", "+append", "bo-example-gp.png", "bo-example-mes.png", "bo-example-gp-mes.png"])

# Plot the MES acquisition function
mes = simple_mes(gp_model)
plt.plot(x, mes, 'r-', label='MES')
plt.xlabel('x')
plt.ylabel('MES(x)')
plt.title('MES Acquisition Function')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('bo-example-mes.png')

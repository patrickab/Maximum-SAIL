import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms import Standardize, Normalize, ChainedInputTransform
from botorch.models.transforms.input import InputStandardize
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood


# Define the objective function
def objective_function(x):
    return 0.5 * x**2 * np.sin(x)

# Define the search space
lower_bound = -10.0
upper_bound = 10.0
space = [(lower_bound, upper_bound)]  # Bounds for x

# Generate random design points
n_design_points = 20
X_design = np.random.uniform(low=-10, high=10, size=(n_design_points, 1))

# Evaluate objective function at design points
Y_design = [objective_function(x) for x in X_design]

# Plot the design points
x = np.linspace(-10, 10, 1000)
y = objective_function(x)
plt.plot(x, y, 'r-', label='Objective Function')
plt.scatter(X_design, Y_design, color='blue', marker='o', label='Design Points')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Design Points Evaluated with Uniform Random Sampling')
plt.legend()
plt.grid(True)
plt.savefig('bo-example-objective.png')

plt.close()

def fit_gp_model(x_sol, y_obj):
    x_tensor = torch.tensor(x_sol, dtype=torch.float64)
    y_tensor = torch.tensor(y_obj, dtype=torch.float64)

    l_bound = torch.tensor([lower_bound], dtype=torch.float64)
    u_bound = torch.tensor([upper_bound], dtype=torch.float64)
    bounds = torch.stack([l_bound, u_bound])

    gp_model = SingleTaskGP(x_tensor, y_tensor)

    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

    fit_gpytorch_model(mll)

    return gp_model

# Fit a gaussian process model using botorch to the design points, plot the mean and the 95% confidence interval
gp_model = fit_gp_model(X_design, Y_design)
# Plot the model and its credible intervals
x = np.linspace(-10, 10, 1000)
x_tensor = torch.tensor(x, dtype=torch.float64).view(-1, 1)
with torch.no_grad():
    f_mean = gp_model(x_tensor).mean
    f_std = gp_model(x_tensor).stddev

plt.plot(x, f_mean, 'r-', label='GP Mean')
plt.fill_between(x, f_mean - 1.96 * f_std, f_mean + 1.96 * f_std, alpha=0.2, label='95% Confidence Interval')
plt.scatter(X_design, Y_design, color='blue', marker='o', label='Design Points')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('GP Model and 95% Confidence Interval')
plt.legend()
plt.grid(True)
plt.savefig('gp_model.png')

import torch
import botorch
from botorch.optim import optimize_acqf


def acquisition_mes(archive, gpModel, genome):

    X = torch.tensor(genome)                                                # Convert genome to tensor
    acq_fitness  = botorch.acquisition.qMaxValueEntropy(gpModel, genome)    # Evaluate fitness using MES // qMaxValueEntropy supports "n x Dim" genomes
    acq_fitness = acq_fitness.detach().numpy()                              # Convert fitness tensor to a numpy array (necessary once genome is "n x Dim" matrice)

    # Define the BoTorch model using the fitness values
    model = your_custom_model(X, fitness)

    # Fit the BoTorch model
    mll = fit_gpytorch_model(model)

    # Define the acquisition function
    acq_function = qMaxValueEntropy(model)

    # Optimize the acquisition function to find the next query point
    candidate, _ = optimize_acqf(
        acq_function,
        bounds=your_custom_bounds,  # Specify the bounds for each feature
        q=1,  # Number of candidates to sample
    )

    # Convert the candidate tensor to a numpy array
    candidate = candidate.detach().numpy()

    acq_behavior = [drag[0], lift[0]]

    return acq_fitness, acq_behavior
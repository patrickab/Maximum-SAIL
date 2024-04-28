import numpy as np
import matplotlib.pyplot as plt

stdev = 0.2
n_mutants = 1000
bounds = np.array([[1,3],[3,7],[2,4]])

def generate_gaussian_mutants(x):

    rng = np.random.default_rng()

    scaled_noise = rng.normal(
        scale=np.abs(stdev * (bounds[:,1] - bounds[:,0])), 
        size=(n_mutants, 3))

    mutants = np.tile(x, (n_mutants, 1)) + scaled_noise
    mutants = np.clip(mutants, bounds[:,0], bounds[:,1])

    return mutants

x = np.array([2, 5, 3])  # Example point
mutants = generate_gaussian_mutants(x)

# Plotting the 3D space and mutants
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the mutants
ax.scatter(mutants[:, 0], mutants[:, 1], mutants[:, 2], c='black', alpha=0.5)

# Plotting the example point
ax.scatter(x[0], x[1], x[2], c='red', s=100)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Gaussian Mutants   sigma={stdev}   n={n_mutants}')

# Set the axes limits
ax.set_xlim(bounds[0])
ax.set_ylim(bounds[1])
ax.set_zlim(bounds[2])

plt.show()

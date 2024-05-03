import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.pyplot as plt

sigma_mutants = 0.2
n_mutants = 1500
cell_bounds = np.array([[1, 3], [3, 7], [2, 4]])
mutant_cellrange = 0.5

# Calculate the range for each cell_bound dimension
cell_range = cell_bounds[:, 1] - cell_bounds[:, 0]

# Modify the cell_bounds based on the range and mutant_cellrange
mutant_cellbounds = cell_bounds.astype(float)
mutant_cellbounds[:, 0] -= cell_range * mutant_cellrange
mutant_cellbounds[:, 1] += cell_range * mutant_cellrange


def generate_gaussian_mutants(x):

    rng = np.random.default_rng()

    scaled_noise = rng.normal(
        scale=np.abs(sigma_mutants * (mutant_cellbounds[:, 1] - mutant_cellbounds[:, 0])),
        size=(n_mutants, 3))

    mutants = np.tile(x, (n_mutants, 1)) + scaled_noise
    mutants = np.clip(mutants, mutant_cellbounds[:, 0], mutant_cellbounds[:, 1])

    return mutants


x = np.array([2, 5, 3])  # Example point
mutants = generate_gaussian_mutants(x)

# Plotting the 3D space and mutants
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

vertices = np.array([[x, y, z] for x in cell_bounds[0] for y in cell_bounds[1] for z in cell_bounds[2]])
lines = [[vertix_i, vertix_j] for vertix_i in vertices for vertix_j in vertices if np.count_nonzero(vertix_i - vertix_j) == 1]

line_collection = Line3DCollection(lines, colors='red')
ax.add_collection(line_collection)

# Plotting the mutants
ax.scatter(mutants[:, 0], mutants[:, 1], mutants[:, 2], c='black', alpha=0.5)

# Plotting the example point
ax.scatter(x[0], x[1], x[2], c='red', s=100)


# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title(f'Gaussian Mutants   sigma={sigma_mutants}   n={n_mutants}   mutant_cellrange={mutant_cellrange}')

# Set the axes limits
ax.set_xlim(mutant_cellbounds[0])
ax.set_ylim(mutant_cellbounds[1])
ax.set_zlim(mutant_cellbounds[2])

plt.show()

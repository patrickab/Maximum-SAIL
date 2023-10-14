#### Import Foreign Packages ###
import os
import numpy as np
from numpy import tan, sqrt

### Import Custom Scripts ###
from utils.pprint_nd import pprint

### Global Variables ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
BATCH_SIZE = config.BATCH_SIZE
N_XY_COORDINATES = config.N_XY_COORDINATES

def export_parsec_coordinates(upper_xy, lower_xy):
    """
    Writes PARSEC-encoded coordinates in 'airfoil_{i}.dat'

    input:      upper & lower coordinates
    output:     .dat file containing coordinates
    """

    valid_indices = []

    for index in range(BATCH_SIZE):
        if os.path.exists("airfoil_{index}.dat"):
            os.remove("airfoil_{index}.dat")

        if os.path.exists("airfoil_{index}.log"):
            os.remove("airfoil_{index}.log")

    surface_batch = np.empty(0)
    for index in range(upper_xy.shape[0]):

        if upper_xy.shape[0] != BATCH_SIZE:
            ValueError(f'EXPORT PARSEC: upper_xy.shape[0] != BATCH_SIZE')

        # Check if polynomials intersect
        upper_i_xy = upper_xy[index,:,:]
        lower_i_xy = lower_xy[index,:,:]
        upper_y = upper_i_xy[:,1]
        lower_y = lower_i_xy[:,1]
        delta_y = upper_y - lower_y
        is_unvalid_airfoil = np.any(delta_y < 0) # returns True if polynomials dont intersect

        if not is_unvalid_airfoil:

            valid_indices.append(index)
            stacked_xy = np.vstack((upper_i_xy, np.flip(lower_i_xy, axis=0)))
            np.savetxt(f'airfoil_{index}.dat', stacked_xy, fmt='%f', delimiter=' ', newline='\n', header=f'airfoil_{index}\n', comments='')

            # Calculate the surface area using the trapezoidal rule in a vectorized manner
            dx = 1/N_XY_COORDINATES
            y_avg = (np.abs(stacked_xy[1:,1]) + np.abs(stacked_xy[:-1,1])) / 2.0
            surface = np.sum(dx * y_avg)
            surface_batch = np.append(surface_batch, surface)

        else:
            with open(f'airfoil_{index}.dat', 'w') as f:
                f.write(f'airfoil_{index}\n\n')
                f.write("Invalid Airfoil\n")

    return np.array(valid_indices), surface_batch

def generate_parsec_coordinates(samples, xte=1.0): # 'x trailing edge'
    """
    Generates Polynomial (x,y) coordinates for sample genomes
    
    input:      samples (scaled to solution space boundaries)
    output:     .txt file with (x,y) coordinates

    returns:    valid_indices (indices of valid airfoils)
                surface_batch (surface area of valid airfoils)
    """

    if samples.shape[0] != BATCH_SIZE:
        ValueError(f'GENERATE PARSEC: samples.shape[0] != BATCH_SIZE')  
    
    n_samples = samples.shape[0]
    valid_indices = np.empty(0)
    surface_batch = np.empty(0)
    
    if n_samples != BATCH_SIZE:
        Warning(f'GENERATE PARSEC: n_samples != BATCH_SIZE')
        print(f"    n_samples: {n_samples}")
        if n_samples == 0:
            return valid_indices, surface_batch

    upper_polynomial_coefficients, lower_polynomial_coefficients = generate_polynomial_coefficients(samples)

    vec_upper_y_solutions = np.vectorize(upper_y_solutions, signature='(6)->(160,2)')
    vec_lower_y_solutions = np.vectorize(lower_y_solutions, signature='(6)->(160,2)')

    upper_xy = vec_upper_y_solutions(upper_polynomial_coefficients)
    lower_xy = vec_lower_y_solutions(lower_polynomial_coefficients)

    valid_indices, surface_batch = export_parsec_coordinates(upper_xy, lower_xy)

    return valid_indices, surface_batch


def generate_polynomial_coefficients(samples):    # PARSEC API: https://github.com/dqsis/parsec-airfoils/blob/master/parsecexport.py
    """
    Solves Linear Equation System
    
    input:      samples (scaled to solution space boundaries)
    output:     .txt file with (x,y) coordinates
    """

    # Parallelize element-wise operations & define I/O shape of vectorized function
    vec_generate_polynomial_terms  = np.vectorize(generate_polynomial_terms, signature='()->(6,6)')
    vec_get_upper_y_vector = np.vectorize(get_upper_y_vector, signature='(7)->(6)')
    vec_get_lower_y_vector = np.vectorize(get_lower_y_vector, signature='(7)->(6)')

    # how can i ensure that my sample array

    # Calculate Linear Equation Systems    (BATCH_SIZE x 6 x 6)
    x_up_matrices = vec_generate_polynomial_terms(samples[:,1].ravel())
    x_low_matrices = vec_generate_polynomial_terms(samples[:,5].ravel())

    # Z position / Thickness of Trailing edge ('z_te' 'dz_te' in seedpaper)
    dz_trailing_edge = np.full(samples.shape[0], 0.001)
    z_trailing_edge  = samples[:,8]

    # Z position of trailing edge
    z_up  = samples[:,2]
    z_low = samples[:,6]
    
    # Trailing edge angles =('a_te b_te' in seedpaper)
    alpha_up  = samples[:,9]
    alpha_low = samples[:,10]

    # Curvature of suction/pressure side ('z_XXup z_XXlo' in seedpaper)
    upper_curvature =  samples[:,3]
    lower_curvature = -samples[:,7]

    # Leading Edge Radius of suction/pressure
    upper_le_radius = samples[:,0]
    lower_le_radius = samples[:,4]

    # Define Parameter Matrix for vec_get_upper_y_vector
    upper_args = (np.array([z_trailing_edge, dz_trailing_edge, z_up, alpha_up, alpha_low, upper_curvature, upper_le_radius])).T
    lower_args = (np.array([z_trailing_edge, dz_trailing_edge, z_low, alpha_up, alpha_low, lower_curvature, lower_le_radius])).T

    # Calculate Solution Vectors for Linear Equation Systems
    b_up = vec_get_upper_y_vector(upper_args)
    b_low = vec_get_lower_y_vector(lower_args)

    # Solve Linear Equation systems
    upper_polynomial_coefficients = np.linalg.solve(x_up_matrices, b_up)
    lower_polynomial_coefficients = np.linalg.solve(x_low_matrices, b_low)

    return upper_polynomial_coefficients, lower_polynomial_coefficients


def upper_y_solutions(upper_polynomial_parameters, n_pts=N_XY_COORDINATES):
    """Evaluates Polynomial Equation for n_pts points (for a single set of parameters)"""

    spacing = np.linspace(0, 1, n_pts)

    # Reverse the array for upper y coordinates (for some reason)
    x_upper_coordinates = spacing[::-1]
    x_upper_coordinates_matrix = (np.tile(x_upper_coordinates, (6, 1)))

    upper_parameters = upper_polynomial_parameters
    upper_parameters = upper_parameters.reshape(-1,1)

    # Rows contain all f(x1), ..., f(x_npts)
    # Columns contain f(x1,p1), ... , f(x1,p6)
    upper_parameters = upper_polynomial_parameters.reshape(-1,1)
    pwrs = np.array([1/2, 3/2, 5/2, 7/2, 9/2, 11/2])

    x_pow = np.power(x_upper_coordinates_matrix.T, pwrs)

    upper_solution_term = upper_polynomial_parameters * x_pow

    y_upper_solutions = np.sum(upper_solution_term, axis=1)

    upper_xy = np.array(list(zip(x_upper_coordinates, y_upper_solutions)))

    return upper_xy


def lower_y_solutions(lower_polynomial_parameters, n_pts=N_XY_COORDINATES):
    """Evaluates Polynomial Equation for n_pts points (for a single set of parameters)"""

    spacing = np.linspace(0, 1, n_pts)

    # Reverse the array for upper y coordinates (for some reason)
    x_lower_coordinates = spacing[::-1]
    x_lower_coordinates_matrix = (np.tile(x_lower_coordinates, (6, 1)))

    pwrs = np.array([1/2, 3/2, 5/2, 7/2, 9/2, 11/2])

    x_pow = np.power(x_lower_coordinates_matrix.T, pwrs)

    lower_solution_term = lower_polynomial_parameters * x_pow

    y_lower_solutions = -np.sum(lower_solution_term, axis=1)

    lower_xy = np.array(list(zip(x_lower_coordinates, y_lower_solutions)))

    return lower_xy


def get_upper_y_vector(args):
    """Calculates Solution Values for Linear Equations"""
    z_trailing_edge, dz_trailing_edge, z_up, alpha_up, alpha_low, upper_curvature, upper_le_radius = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
    alpha_up = np.deg2rad(alpha_up)
    alpha_low = np.deg2rad(alpha_low)
    return np.array([ z_trailing_edge + (dz_trailing_edge/2),  z_up, tan(alpha_up - (alpha_low/2)), 0, upper_curvature, sqrt(2*upper_le_radius)])


def get_lower_y_vector(args):
    """Calculates Solution Values for Linear Equations"""
    z_trailing_edge, dz_trailing_edge, z_low, alpha_up, alpha_low, lower_curvature, lower_le_radius = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
    alpha_up = np.deg2rad(alpha_up)
    alpha_low = np.deg2rad(alpha_low)

    return np.array([-z_trailing_edge + (dz_trailing_edge/2), -z_low, tan(alpha_up + (alpha_low/2)), 0, lower_curvature, sqrt(2*lower_le_radius)])


def generate_polynomial_terms(sample):
    """Generates Set of Linear Equations"""

    x = np.asarray(sample)

    x_matrix = np.array([
        np.array(np.ones(6)),
        x**((np.arange(1, 12, 2)/2)),
        (np.arange(1, 12, 2) / 2),
        np.array([(1/2) *x**(-1/2), (3/2)*x**(1/2) , (5/2) *x**(3/2), (7/2) *x**(5/2), (9/2) *x**(7/2), (11/2)*x**(9/2)]),
        np.array([(-1/4)*x**(-3/2), (3/4)*x**(-1/2), (15/4)*x**(1/2), (35/4)*x**(3/2), (53/4)*x**(5/2), (99/4)*x**(7/2)]),
        np.array([1, 0, 0, 0, 0, 0])])
    
    return x_matrix
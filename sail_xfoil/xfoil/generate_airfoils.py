import os
import numpy as np
from numpy import tan, sqrt

### Custom Scripts ###
from utils.pprint_nd import pprint, pprint_nd

### Global Variables ###
from config.config import Config
config = Config(os.path.join(os.path.dirname(__file__), '../config', 'config.ini'))
PARALLEL_BATCH_SIZE = config.PARALLEL_BATCH_SIZE


### ADD EXPONENTIAL TERMS TO MATRICES
### PRODUCE NEGATIVE Y VALUES
    ### write coordinates into text file
        ### pass coordinates to simulate_objective()
            ### [XFoil]


def generate_parsec_coordinates(samples, xte=1.0): # 'n points' and 'x trailing edge'
    """
    Generates Polynomial (x,y) coordinates for samples genomes
    
    input: samples (scaled to solution space boundaries)
    output: .txt file with (x,y) coordinates
    """

    print("\n\ngenerating airfoils...\n\n")

    upper_polynomial_coefficients, lower_polynomial_coefficients = generate_parsec_polynomials(samples)

    vec_upper_y_solutions = np.vectorize(upper_y_solutions, signature='(6)->(200)')
    vec_lower_y_solutions = np.vectorize(lower_y_solutions, signature='(6)->(200)')

    # Shape = (PARALLEL_BATCH_SIZE, n_pts)
    upper_y = vec_upper_y_solutions(upper_polynomial_coefficients)
    lower_y = vec_lower_y_solutions(lower_polynomial_coefficients)

    pprint(upper_y)
    pprint(lower_y)


def generate_parsec_polynomials(samples):    # PARSEC API: https://github.com/dqsis/parsec-airfoils/blob/master/parsecexport.py
    """Solves Linear Equation System"""

    # Parallelize element-wise operations & define I/O shape of vectorized function
    vec_generate_polynomial_terms  = np.vectorize(generate_polynomial_terms, signature='()->(6,6)')
    vec_get_upper_y_sol_vector = np.vectorize(get_upper_y_vector, signature='(7)->(6)')
    vec_get_lower_y_sol_vector = np.vectorize(get_lower_y_vector, signature='(7)->(6)')

    # Calculate Vector of Linear Equation Systems
    x_up_matrices = vec_generate_polynomial_terms(samples[:,1].ravel())
    x_low_matrices = vec_generate_polynomial_terms(samples[:,5].ravel())

    # Z position / Thickness of Trailing edge ('z_te' 'dz_te' in seedpaper)
    z_trailing_edge  = samples[:,8]
    dz_trailing_edge = samples[:,9]

    # Z position of trailing edge
    z_up  = samples[:,2]
    z_low = samples[:,6]
    
    # Trailing edge angles =('a_te b_te' in seedpaper)
    alpha_up  = samples[:,10]
    alpha_low = samples[:,11]

    # Curvature of suction/pressure side ('z_XXup z_XXlo' in seedpaper)
    upper_curvature =  samples[:,3]
    lower_curvature = -samples[:,7]

    # Leading Edge Radius of suction/pressure
    upper_le_radius = samples[:,0]
    lower_le_radius = samples[:,4]

    # Define Parameter Matrix for vec_get_upper_y_sol_vector
    upper_args = (np.array([z_trailing_edge, dz_trailing_edge, z_up, alpha_up, alpha_low, upper_curvature, upper_le_radius])).T
    lower_args = (np.array([z_trailing_edge, dz_trailing_edge, z_low, alpha_up, alpha_low, lower_curvature, lower_le_radius])).T

    # Calculate Solution Vectors for Linear Equation Systems
    b_up = vec_get_upper_y_sol_vector(upper_args)
    b_low = vec_get_lower_y_sol_vector(lower_args)

    # Solve Linear Equation systems
    upper_polynomial_coefficients = np.linalg.solve(x_up_matrices, b_up)
    lower_polynomial_coefficients = np.linalg.solve(x_low_matrices, b_low)

    return upper_polynomial_coefficients, lower_polynomial_coefficients


def upper_y_solutions(upper_polynomial_parameters, n_pts=200):
    """Evaluates Polynomial Equation for 200 points (for a single set of parameters)"""

    spacing = np.linspace(0, 1, n_pts)

    # Reverse the array for upper y coordinates (for some reason)
    x_upper_coordinates = spacing[::-1]
    x_upper_coordinates_matrix = (np.tile(x_upper_coordinates, (6, 1)))

    upper_parameters = upper_polynomial_parameters
    upper_parameters = upper_parameters.reshape(-1,1)

    # Rows contain all f(x1), ..., f(x_npts)
    # Columns contain f(x1,p1), ... , f(x1,p6)
    upper_parameters = upper_polynomial_parameters.reshape(-1,1)

    y_upper_polynomial_terms = (upper_parameters*(x_upper_coordinates_matrix)).T
    y_upper_coordinates = np.sum(y_upper_polynomial_terms, axis = 1)

    return y_upper_coordinates


def lower_y_solutions(lower_polynomial_parameters, n_pts=200):
    """Evaluates Polynomial Equation for 200 points (for a single set of parameters)"""

    x_lower_coordinates = np.linspace(0, 1, n_pts)

    x_lower_coordinates_matrix = (np.tile(x_lower_coordinates, (6, 1)))

    # Rows contain all f(x1), ..., f(x_npts)
    # Columns contain f(x1,p1), ... , f(x1,p6)
    lower_parameters = lower_polynomial_parameters.reshape(-1,1)

    y_lower_polynomial_terms = (lower_parameters*(x_lower_coordinates_matrix)).T
    y_lower_coordinates = np.sum(y_lower_polynomial_terms, axis = 1)

    return y_lower_coordinates


def get_upper_y_vector(args):
    """Calculates Solution Values for Linear Equations"""
    z_trailing_edge, dz_trailing_edge, z_up, alpha_up, alpha_low, upper_curvature, upper_le_radius = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
    return np.array([ z_trailing_edge + (dz_trailing_edge/2),  z_up, tan(alpha_up - (alpha_low/2)), 0, upper_curvature, sqrt(2*upper_le_radius)])


def get_lower_y_vector(args):
    """Calculates Solution Values for Linear Equations"""
    z_trailing_edge, dz_trailing_edge, z_low, alpha_up, alpha_low, lower_curvature, lower_le_radius = args[0], args[1], args[2], args[3], args[4], args[5], args[6]
    return np.array([-z_trailing_edge + (dz_trailing_edge/2), z_low, tan(alpha_up + (alpha_low/2)), 0, lower_curvature, sqrt(2*lower_le_radius)])


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

import os
import numpy
import numpy as np
from queue import Queue, Empty
import inspect

N_XY_COORDINATES = 65

def calculate_rae2822_surface_area():

    # def calculate_area_penalty(surface_area):
    #     return 1 - numpy.absolute(surface_area - base_area)


    data = np.loadtxt(f'rae2822.dat', skiprows=2)

    upper_xy = np.array([data[0:N_XY_COORDINATES, 0], numpy.absolute(data[0:N_XY_COORDINATES, 1]) ]).T # note: this function does not consider, that y values can alterate between positive & negative values
    lower_xy = np.array([data[N_XY_COORDINATES:, 0], numpy.absolute(data[N_XY_COORDINATES:, 1])]).T # note: this function does not consider, that y values can alterate between positive & negative values

    surface_area = 0.0

    # Calculate the surface area using the trapezoidal rule
    for i in range(N_XY_COORDINATES-1):

        upper_dx = (upper_xy[i+1, 0] - upper_xy[i, 0])
        lower_dx = (lower_xy[i+1, 0] - lower_xy[i, 0])

        upper_y_avg = (upper_xy[i + 1, 1] + upper_xy[i, 1]) / 2.0
        lower_y_avg = (lower_xy[i + 1, 1] + lower_xy[i, 1]) / 2.0


        surface_area += numpy.absolute(upper_dx * upper_y_avg) + numpy.absolute(lower_dx * lower_y_avg)

    return surface_area

# calculate drag by taking rae2822 coordinate data from
# http://airfoiltools.com/airfoil/details?airfoil=rae69ck-il ('selig format dat file')


def pprint(variable):

    # Get the calling frame
    frame = inspect.currentframe().f_back

    # Find the variable name that was passed as an argument to the function
    variable_name = None
    for name, value in frame.f_locals.items():
        if value is variable:
            variable_name = name
            break

    if isinstance(variable, np.ndarray):
        shape_string = str(variable.shape)

    print("\n Name: " + str(variable_name))
    print(" Type: " + str(type(variable)))
    print("Shape: " + str(variable.shape))
    print(variable)

    return


def pprint_nd(matrix1, matrix2, matrix3=None):

    frame = inspect.currentframe().f_back

    str_matrix1 = None
    for name, value in frame.f_locals.items():
        if value is matrix1:
            str_matrix1 = name
            break
        
    str_matrix2 = None
    for name, value in frame.f_locals.items():
        if value is matrix2:
            str_matrix2 = name
            break

    if matrix3 is not None:
        str_matrix3 = None
        for name, value in frame.f_locals.items():
            if value is matrix3:
                str_matrix3 = name
                break
            array_touples = [(matrix1, str_matrix1), (matrix2, str_matrix2), (matrix3, str_matrix3)]
    
    else:
        array_touples = [(matrix1, str_matrix1), (matrix2, str_matrix2)]
    
    format_string = ""
    
    for matrix, name in array_touples:
        if matrix is not None:
            type_matrix = str((type(matrix)))
            shape_matrix = str(matrix.shape)
            decorator = f"\n\n\nName: {name} {shape_matrix}, Type: {type_matrix}\n\n"
            format_string += decorator
            
            if matrix.ndim == 1:
                formatted_row = "\t".join([f"{value:.8f}" for value in matrix])
                format_string += formatted_row + "\n"
            else:
                for row in matrix:
                    formatted_row = "\t".join([f"{value:.8f}" for value in row])
                    format_string += formatted_row + "\n"
    
    print(format_string)


if __name__ == "__main__":

    calculate_rae2822_drag()
    surface = calculate_rae2822_surface_area()

    print("surface: " + str(surface))

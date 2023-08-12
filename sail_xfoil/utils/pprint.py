import inspect
import numpy as np

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

    print("\n Name: " + variable_name)
    print(" Type: " + str(type(variable)))
    print("Shape: " + str(variable.shape))

    print(variable_name + ": ")
    print(variable)

    return
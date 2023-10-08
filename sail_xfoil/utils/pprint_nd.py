import numpy as np
import inspect


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
    if isinstance(variable, np.ndarray):
        print("Shape: " + str(variable.shape))
    print(variable)
    print()
    return


def pprint_fstring(ndarray1, ndarray2 = None, ndarray3 = None):

    frame = inspect.currentframe().f_back

    name_ndarray1 = None
    for name, value in frame.f_locals.items():
        if value is ndarray1:
            name_ndarray1 = name
            break
    
    name_ndarray2 = None
    for name, value in frame.f_locals.items():
        if value is ndarray2:
            name_ndarray2 = name
            break

    if ndarray3 is not None:
        name_ndarray3 = None
        for name, value in frame.f_locals.items():
            if value is ndarray3:
                name_ndarray3 = name
                break

    if ndarray2 is None and ndarray3 is None:
        fstring_names = f"{name_ndarray1}:"
        format_string = "\n".join(["\t".join(map(str, row)) for row in ndarray1])
        format_string = "\n".join(["\t".join([f"{value:.4f}" for value in row]) for row in ndarray1])
    if ndarray3 is None:
        fstring_names = f"{name_ndarray1}:\t{name_ndarray2}:"
        stacked_arrays = np.column_stack((ndarray1, ndarray2))
        format_string = "\n".join(["\t\t".join(map(str, row)) for row in stacked_arrays])
        format_string = "\n".join(["\t\t\t".join([f"{value:.4f}" if type(value)!=int else value for value in row]) for row in stacked_arrays])
    else:
        fstring_names = f"{name_ndarray1}:\t{name_ndarray2}:\t{name_ndarray3}:"
        stacked_arrays = np.column_stack((ndarray1, ndarray2, ndarray3))
        format_string = "\n".join(["\t\t".join(map(str, row)) for row in stacked_arrays])
        format_string = "\n".join(["\t\t\t".join([f"{value:.4f}" for value in row]) for row in stacked_arrays])


    print("\n" + fstring_names + "\n" + format_string + "\n")



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
                formatted_row = "\t".join([f"{value:.4f}" for value in matrix])
                format_string += formatted_row + "\n"
            else:
                for row in matrix:
                    formatted_row = "\t".join([f"{value:.4f}" for value in row])
                    format_string += formatted_row + "\n"
    
    print(format_string)

def pprint_nd_array(array1,array2,array3=None):

    frame = inspect.currentframe().f_back

    print("\n\nenter print_nd_array...\n\n")
    
    str_array1 = None
    for name, value in frame.f_locals.items():
        if value is array1:
            str_array1 = str(name)
            break

    str_array2 = None
    for name, value in frame.f_locals.items():
        if value is array2:
            str_array2 = str(name)
            break

    if array3 is not None:
        str_array3 = None
        for name, value in frame.f_locals.items():
            if value is array3:
                str_array3 = name
                array1 = array1.reshape(-1,1)
                array2 = array2.reshape(-1,1)
                array3 = array3.reshape(-1,1)
                arrayName_touples = [(array1, str_array1), (array2, str_array2), (array3, str_array3)]
                
                print("pprint_nd")
                print(arrayName_touples)
                break
            else:
                arrayName_touples = [(array1, str_array1), (array2, str_array2)]
    

    for array, name in arrayName_touples:
        if array is not None:
            if array.ndim == 1 and array.shape != (array.size, 1):
                array = array.reshape(-1, 1)


    format_string = ""
    
    for array, name in arrayName_touples:
        if array is not None:
            type_array = str(type(array))
            shape_array = str(array.shape)
            decorator = f"\n\n\nName: {name} {shape_array}, Type: {type_array}\n\n"
            format_string += decorator

            #print(format_string)
            
            if array.ndim == 1:
                formatted_row = "\t".join([f"{value:.4f}" for value in array])
                format_string += formatted_row + "\n"
            else:
                for row in array:
                    formatted_row = "\t".join([f"{value:.4f}" for value in row])
                    format_string += formatted_row + "\n"


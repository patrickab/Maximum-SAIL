import numpy as np
import inspect


# def pprint(variable):

#     # Get the calling frame
#     frame = inspect.currentframe().f_back

#     # Find the variable name that was passed as an argument to the function
#     variable_name = None
#     for name, value in frame.f_locals.items():
#         if value is variable:
#             variable_name = name
#             break

#     if isinstance(variable, np.ndarray):
#         shape_string = str(variable.shape)

#     print("\n Name: " + str(variable_name))
#     print(" Type: " + str(type(variable)))
#     if isinstance(variable, np.ndarray):
#         print("Shape: " + str(variable.shape))
#     print(variable)
#     print()
#     return


def pprint(array1, array2=None, array3=None):

    def pprint1(array1):

        array1 = np.vstack(array1)
        print(array1)


    def pprint2(array1, array2):

        array1 = np.vstack(array1)
        array2 = np.vstack(array2)

        for sol, obj in zip(array1, array2):
            print(sol, obj, sep="\t")

    def pprint3(array1, array2, array3):

        array1 = np.vstack(array1)
        array2 = np.vstack(array2)
        array3 = np.vstack(array3)


        for sol, obj, bhv in zip(array1, array2, array3):
            print(sol, obj, bhv, sep="\t")


    frame = inspect.currentframe().f_back
    name_array1 = None
    name_array2 = None
    name_array3 = None

    for name, value in frame.f_locals.items():
        if value is array1:
            name_array1 = name
        if value is array2:
            name_array2 = name
        if value is array3:
            name_array3 = name
            break

    if array2 is None and array3 is None:

        print(f"{name_array1}:")
        pprint1(array1)
        return

    if array3 is None:

        rowlength1 = np.vstack(array1).shape[1]
        tabs1 = "\t" * (rowlength1//2)

        rowlength2 = np.vstack(array2).shape[1]
        tabs2 = "\t" * (rowlength2//2)

        print(f"{tabs1}{name_array1}{tabs1+tabs2}{name_array2}:")
        pprint2(array1, array2)
        return

    rowlength1 = np.vstack(array1).shape[1]
    tabs1 = "\t" * (rowlength1//2)

    rowlength2 = np.vstack(array2).shape[1]
    tabs2 = "\t" * (rowlength2//2)

    tab = "\t" 

    print(f"{tabs1}{name_array1}{tabs1+tab}{name_array2}{tabs2+tab}{name_array3}:")
    pprint3(array1, array2, array3)
    return
"""Pretty print numpy arrays in a more readable format."""


import numpy as np
import inspect


def pprint(array1, array2=None, array3=None):

    def pprint1(array1):

        array1 = np.vstack(array1)
        print(array1)


    def pprint2(array1, array2, max_name_length):

        array1 = np.vstack(array1)
        array2 = np.vstack(array2)

        for sol, obj in zip(array1, array2):    
            sol_padded = ' '.join(f'{val:.4f}'.ljust(max_name_length) for val in sol)
            obj_padded = ' '.join(f'{val:.4f}'.ljust(max_name_length) for val in obj)
            print(f"{sol_padded}  {obj_padded}")
        print("\n")

    def pprint3(array1, array2, array3):

        array1 = np.vstack(array1)
        array2 = np.vstack(array2)
        array3 = np.vstack(array3)


        for sol, obj, bhv in zip(array1, array2, array3):
            print(sol, obj, bhv, sep="\t\t")


    frame = inspect.currentframe().f_back

    name_array1 = None
    for name, value in frame.f_locals.items():
        if value is array1:
            name_array1 = name
            break

    name_array2 = None
    for name, value in frame.f_locals.items():
        if value is array2:
            name_array2 = name
            break
    
    name_array3 = None
    for name, value in frame.f_locals.items():
        if value is array3:
            name_array3 = name
            break

    if array2 is None and array3 is None:

        print(f"{name_array1}:")
        pprint1(array1)
        return

    if array3 is None:

        max_name_length = max(len(name_array1), len(name_array2))
    
        # Pad the names with spaces to align them properly
        name1_padded = name_array1.ljust(max_name_length)
        name2_padded = name_array2.ljust(max_name_length)
    
        # Print the names above the arrays
        print(f"{name1_padded}  {name2_padded}:")

        pprint2(array1=array1, array2=array2, max_name_length=max_name_length)
        return

    if array1.size == 0 and array2.size == 0 and array3.size == 0:
        print("Empty arrays")
        return
    
    rowlength1 = np.vstack(array1).shape[1]
    tabs1 = "\t\t" * (rowlength1//2) if rowlength1 > 1 else ""

    rowlength2 = np.vstack(array2).shape[1]
    tabs2 = "\t\t" * (rowlength2//2) if rowlength1 > 1 else "\t\t"

    tab = "\t\t" 

    print(f"{tabs1}{name_array1}{tabs1+tab}{name_array2}{tabs2+tab}{name_array3}:")
    pprint3(array1, array2, array3)
    print("\n")
    return
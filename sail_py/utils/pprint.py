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

    print("\nType " + variable_name + ": " + str(type(variable)))
    print(variable_name + ": ")
    print(variable)
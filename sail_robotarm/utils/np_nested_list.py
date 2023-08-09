import numpy as np

def np_nested_list(batch):
    """"Convert 'n x 1' numpy-batch to 'n x dim'"""
    nested_batch = [[np.array(element) for element in row] for row in batch]
    return nested_batch
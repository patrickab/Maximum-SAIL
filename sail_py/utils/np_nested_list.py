import numpy as np

def np_nested_list(batch):
    """"Convert n x 1 batch to n x dim"""
    
    nested_list = [[np.array(element) for element in row] for row in batch]

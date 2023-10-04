import os

### Global Parameters ###
from config import Config
config = Config(os.path.join(os.path.dirname(__file__), '..', 'config.ini'))
SOL_VALUE_RANGE = config.SOL_VALUE_RANGE

def scale_samples(samples):
    for i in range(len(samples)):
        for j in range(len(samples[i])):
            lower_bound, upper_bound = SOL_VALUE_RANGE[j]
            samples[i][j] = samples[i][j] *(upper_bound - lower_bound) + lower_bound

    return samples

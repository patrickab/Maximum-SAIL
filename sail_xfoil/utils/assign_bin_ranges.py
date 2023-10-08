import numpy as np
from ribs.emitters import GaussianEmitter
from ribs.archives import GridArchive

from utils.pprint_nd import pprint

BATCH_SIZE = 10
BHV_NUMBER_BINS = [2,2]
BHV_VALUE_RANGES = [(0,10),(0,10)]

# this script demonstrates the behavior of bin indexing in the grid archive

def main():

    SOL_VALUE_RANGE = [(20,30), (0,10),(0,10),(-30,-20)]
    ranges = np.array(SOL_VALUE_RANGE)

    def uniform_sample():
        uniform_sample = np.random.uniform(ranges[:, 0], ranges[:, 1], 4)
        return uniform_sample

    init_samples = np.array([uniform_sample() for i in range(BATCH_SIZE)])
    init_samples_bhv = init_samples[:,1:3]

    pprint(init_samples)
    pprint(init_samples_bhv)

    obj_archive = GridArchive(
    solution_dim=4,                           # Dimension of solution vector
    dims=BHV_NUMBER_BINS,                       # Discretization of behavioral bins
    ranges=BHV_VALUE_RANGES,                     # Possible values for behavior vector
    qd_score_offset=-600,
    threshold_min = -1,
    seed=1
    )

    obj_archive.add(init_samples, np.ones(BATCH_SIZE), init_samples_bhv)
    print(str(obj_archive.stats.num_elites))

    archive_indices = obj_archive.index_of(init_samples_bhv)
    pprint(archive_indices)

    idx = obj_archive.int_to_grid_index(archive_indices)
    pprint(idx)

    val_rngs = np.empty((BATCH_SIZE,2))
    for i in range(BATCH_SIZE):
        lower_measure_0 = obj_archive.boundaries[0][idx[0]]
        upper_measure_0 = obj_archive.boundaries[0][idx[0]+1]

        print(lower_measure_0)
        print(upper_measure_0)

        lower_measure_1 = obj_archive.boundaries[0][idx[1]]
        upper_measure_1 = obj_archive.boundaries[1][idx[1]]

        val_rng_measure_0 = (lower_measure_0, upper_measure_0)
        val_rng_measure_1 = (lower_measure_1, upper_measure_1)

        print(val_rng_measure_0)
        print(val_rng_measure_1)

        val_rng = np.array([val_rng_measure_0, val_rng_measure_1])

        pprint(val_rng)
        
        val_rngs[i] = val_rng

    pprint(val_rngs)

    for i in range(BATCH_SIZE):
        lower = obj_archive.boundaries[0][idx[0]]
        upper = obj_archive.boundaries[0][idx[0]+1]
        print("lower: " + str(lower))
        print("upper: " + str(upper))

    obj_emitter = [
        GaussianEmitter(
        archive=obj_archive,
        sigma=0.5,
        bounds= np.array(SOL_VALUE_RANGE),
        batch_size=BATCH_SIZE,
        initial_solutions=init_samples, # these solutions are never used, as the archive is never empty - however, specification is required for initializing the GaussianEmitter class
        seed=1
    )]

if __name__ == '__main__':
    main()
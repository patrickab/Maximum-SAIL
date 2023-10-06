from ribs.archives import GridArchive
from ribs.visualize import grid_archive_heatmap
import matplotlib.pyplot as plt
import inspect
import numpy
import csv

SOL_DIMENSION = 11
BHV_NUMBER_BINS = [25,25]
BHV_VALUE_RANGE = [(0.2625,0.6875), (0.0725,0.1875)]

numpy.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=120)

def main():

    comb_obj = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = -1
    )

    comb_pred_error = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = -1
    )

    comb_unverified_obj = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = -1
    )
    
    comb_verified_obj = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = -1
    )

    comb_obj = convert_csv_to_archive('combined_obj.csv', comb_obj)
    comb_pred_error = convert_csv_to_archive('combined_pred_error.csv', comb_pred_error)
    comb_verified_obj = convert_csv_to_archive('combined_verified_obj.csv', comb_verified_obj)
    comb_unverified_obj = convert_csv_to_archive('combined_unverified_obj.csv', comb_unverified_obj)



    visualize_archive(comb_obj)
    visualize_archive(comb_pred_error)
    visualize_archive(comb_unverified_obj)
    visualize_archive(comb_verified_obj)   


def visualize_archive(archive):


    frame = inspect.currentframe().f_back

    archive_name = None
    for name, value in frame.f_locals.items():
        if value is archive:
            archive_name = name
            break

    print(f"Elites in {archive_name}:  " + str(archive.stats.num_elites))

    fig, ax = plt.subplots(figsize=(8, 8))
    grid_archive_heatmap(archive, ax=ax)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(f"Grid Archive Heatmap ({archive_name})")
    fig.savefig(f"{archive_name}_heatmap.png")


def convert_csv_to_archive(filename, archive):

    data = numpy.genfromtxt(filename, delimiter=',', skip_header=1)

    sol = data[:, 4:]
    obj = data[:, 3]
    bhv = data[:, 1:3]


    pprint(sol)
    pprint(obj)
    pprint(bhv)    

    archive.add(sol, obj, bhv)

    return archive


def pprint(variable):

    frame = inspect.currentframe().f_back

    variable_name = None
    for name, value in frame.f_locals.items():
        if value is variable:
            variable_name = name
            break

    if isinstance(variable, numpy.ndarray):
        shape_string = str(variable.shape)

    print("\n Name: " + str(variable_name))
    print(" Type: " + str(type(variable)))
    print("Shape: " + str(variable.shape))
    print(variable)
    print()
    return


if __name__ == '__main__':
    main()
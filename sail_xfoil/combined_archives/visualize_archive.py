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

def main(benchmark_domain: str):

    objective_archive, pred_error_archive, unverified_predictions, verified_predictions = define_archives()

    objective_archive = convert_csv_to_archive(f'combined_obj_{benchmark_domain}.csv', objective_archive)
    pred_error_archive = convert_csv_to_archive(f'combined_pred_error_{benchmark_domain}.csv', pred_error_archive)
    verified_predictions = convert_csv_to_archive(f'combined_verified_obj_{benchmark_domain}.csv', verified_predictions)
    unverified_predictions = convert_csv_to_archive(f'combined_unverified_obj_{benchmark_domain}.csv', unverified_predictions)

    max_obj, max_pred_error, max_verified_obj, max_unverified_obj = determine_elite_argmax("custom")#, "vanilla", "random")

    visualize_archive(objective_archive, benchmark_domain, max_obj)
    visualize_archive(pred_error_archive, benchmark_domain, max_pred_error)
    visualize_archive(verified_predictions, benchmark_domain, max_verified_obj)
    visualize_archive(unverified_predictions, benchmark_domain, max_unverified_obj)   


def determine_elite_argmax(benchmark_domain_1: str, benchmark_domain_2: str = None, benchmark_domain_3: str = None):
    """Determine maximum objectives for all archives in all domains to ensure comparable colorscaling"""

    if benchmark_domain_2 is None:
        benchmark_domain_2 = benchmark_domain_1
    if benchmark_domain_3 is None:
        benchmark_domain_3 = benchmark_domain_1

    benchmark_domains = [benchmark_domain_1, benchmark_domain_2, benchmark_domain_3]
    archives = ["objective_archive", "pred_error_archive", "verified_predictions", "unverified_predictions"]
    max_obj = 0
    max_pred_error = 0
    max_verified_obj = 0
    max_unverified_obj = 0

    for domain in benchmark_domains:
        objective_archive, pred_error_archive, unverified_predictions, verified_predictions = define_archives()
        objective_archive = convert_csv_to_archive(f'combined_obj_{domain}.csv', objective_archive)
        pred_error_archive = convert_csv_to_archive(f'combined_pred_error_{domain}.csv', pred_error_archive)
        verified_predictions = convert_csv_to_archive(f'combined_verified_obj_{domain}.csv', verified_predictions)
        unverified_predictions = convert_csv_to_archive(f'combined_unverified_obj_{domain}.csv', unverified_predictions)


        archive_max_obj = objective_archive.best_elite[1]
        max_obj = max(max_obj, archive_max_obj)

        archive_max_pred_error = pred_error_archive.best_elite[1]
        # very high prediction errors result in bad plots
        if archive_max_pred_error < 1:
            max_pred_error = max(max_pred_error, archive_max_pred_error)
        else:
            max_pred_error = max(max_pred_error, 1)
            print(f"### Max Prediction Error in {domain}: {archive_max_pred_error}")

        archive_max_verified_obj = verified_predictions.best_elite[1]
        max_verified_obj = max(max_verified_obj, archive_max_verified_obj)

        archive_max_unverified_obj = unverified_predictions.best_elite[1]
        max_unverified_obj = max(max_unverified_obj, archive_max_unverified_obj)

    return max_obj, max_pred_error, max_verified_obj, max_unverified_obj


def visualize_archive(archive, benckmark_domain: str, max_obj: float):


    frame = inspect.currentframe().f_back

    archive_name = None
    for name, value in frame.f_locals.items():
        if value is archive:
            archive_name = name
            break

    print(f"Elites in {archive_name}:  " + str(archive.stats.num_elites))

    fig, ax = plt.subplots(figsize=(8, 8))
    grid_archive_heatmap(archive, ax=ax, vmin=0, vmax=max_obj)
    plt.xlabel("X Up")
    plt.ylabel("Z Up")
    plt.title(f"Heatmap: ({benckmark_domain} ,{archive_name})")
    fig.savefig(f"{benckmark_domain}_{archive_name}_heatmap.png")


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


def define_archives():
    objective_archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = -1
    )

    pred_error_archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = -1
    )

    unverified_predictions = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = -1
    )
            
    verified_predictions = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
        qd_score_offset=-600,
        threshold_min = -1
    )

    return objective_archive, pred_error_archive, unverified_predictions, verified_predictions


if __name__ == '__main__':
    main("custom")
    #main("vanilla")
    #main("random")
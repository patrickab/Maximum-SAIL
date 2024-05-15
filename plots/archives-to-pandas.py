import os
import pandas
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from ribs.archives import GridArchive, ArchiveDataFrame
import pandas as pd

benchmark = "botorch_acqf_a58d2ab"
boxplot_title = "Vanilla-SAIL n=1280"

directory = os.path.expanduser(f"~/Maximum-SAIL/data/benchmarks/{benchmark}/csv")
directory = os.path.abspath(directory)

SOL_DIMENSION = 11
OBJ_BHV_NUMBER_BINS = [25,25]
BHV_VALUE_RANGE = [(0.2625,0.6875), (0.0725,0.1875)]


def csv_to_archive(csv_path: str):

    df = pandas.read_csv(csv_path)
    df = ArchiveDataFrame(df)

    archive = GridArchive(
        solution_dim=SOL_DIMENSION,
        dims=OBJ_BHV_NUMBER_BINS,
        ranges=BHV_VALUE_RANGE,
    )

    archive.add(df.solution_batch(), df.objective_batch(), df.measures_batch())
    return archive


np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=120)

obj_archive_path = os.path.join(directory, "obj_archive")
pred_archive_path = os.path.join(directory, "pred_archive")
evaluated_pred_archive_path = os.path.join(directory, "evaluated_pred_archive")

obj_mean = np.empty((0,1), dtype=np.float64)
obj_max = np.empty((0,1), dtype=np.float64)
obj_coverage = np.empty((0,1), dtype=np.float64)

pred_mean = np.empty((0,1), dtype=np.float64)
pred_max = np.empty((0,1), dtype=np.float64)

evaluated_pred_mean = np.empty((0,1), dtype=np.float64)
evaluated_pred_max = np.empty((0,1), dtype=np.float64)
evaluated_pred_coverage = np.empty((0,1), dtype=np.float64)

mse = np.empty((0,1), dtype=np.float64)
mae = np.empty((0,1), dtype=np.float64)
mpe = np.empty((0,1), dtype=np.float64)

# iterate over the runs
for i in range(0, 9):

    obj_filenames = os.listdir(obj_archive_path)
    obj_filename = [filename for filename in obj_filenames if filename.startswith(f"{i}_")]

    pred_filenames = os.listdir(pred_archive_path)
    pred_filename = [filename for filename in pred_filenames if filename.startswith(f"{i}_")]

    evaluated_pred_filenames = os.listdir(evaluated_pred_archive_path)
    evaluated_pred_filename = [filename for filename in evaluated_pred_filenames if filename.startswith(f"{i}_")]

    obj_archive = csv_to_archive(os.path.join(obj_archive_path, obj_filename[0]))
    pred_archive = csv_to_archive(os.path.join(pred_archive_path, pred_filename[0]))
    evaluated_pred_archive = csv_to_archive(os.path.join(evaluated_pred_archive_path, evaluated_pred_filename[0]))

    pred_archive_df = pred_archive.as_pandas(include_solutions=True).sort_values(by=['index'], ascending=True)
    evaluated_pred_archive_df = evaluated_pred_archive.as_pandas(include_solutions=True).sort_values(by=['index'], ascending=True)

    pred_indices = pred_archive_df['index']
    evaluated_pred_indices = evaluated_pred_archive_df['index']
    is_converged_pred = np.isin(pred_indices, evaluated_pred_indices)

    pred_archive_df = pred_archive_df[is_converged_pred]

    pred_archive.clear()
    pred_archive.add(pred_archive_df.solution_batch(), pred_archive_df.objective_batch(), pred_archive_df.measures_batch())

    if not np.all(pred_archive_df.solution_batch() - evaluated_pred_archive_df.solution_batch() == 0):
        RuntimeError("Something went wrong")
        print(pred_archive_df.solution_batch() - evaluated_pred_archive_df.solution_batch())
    
    absolute_error = np.abs(pred_archive_df.objective_batch() - evaluated_pred_archive_df.objective_batch())
    percentual_error = np.mean(np.abs(absolute_error / evaluated_pred_archive_df.objective_batch()))
    mean_percentual_error = np.mean(percentual_error)
    mean_squared_error = np.mean(np.square(absolute_error))
    mean_absolute_error = np.mean(absolute_error)

    mse = np.append(mse, mean_squared_error)
    mae = np.append(mae, mean_absolute_error)
    mpe = np.append(mpe, mean_percentual_error)

    if mean_percentual_error < 0:
        print(mean_percentual_error)

    obj_mean = np.append(obj_mean, obj_archive.stats.obj_mean)
    obj_max = np.append(obj_max, obj_archive.stats.obj_max)
    obj_coverage = np.append(obj_coverage, obj_archive.stats.coverage)

    pred_mean = np.append(pred_mean, pred_archive.stats.obj_mean)
    pred_max = np.append(pred_max, pred_archive.stats.obj_max)

    evaluated_pred_mean = np.append(evaluated_pred_mean, evaluated_pred_archive.stats.obj_mean)
    evaluated_pred_max = np.append(evaluated_pred_max, evaluated_pred_archive.stats.obj_max)
    evaluated_pred_coverage = np.append(evaluated_pred_coverage, evaluated_pred_archive.stats.coverage)

data = {
    'mse': mse,
    'mae': mae,
    'mpe': mpe,
    'obj_mean': obj_mean,
    'obj_max': obj_max,
    'obj_coverage': obj_coverage,
    'pred_mean': pred_mean,
    'pred_max': pred_max,
    'evaluated_pred_mean': evaluated_pred_mean,
    'evaluated_pred_max': evaluated_pred_max,
    'evaluated_pred_coverage': evaluated_pred_coverage
}

df = pd.DataFrame(data)
df.to_csv(f'{benchmark}.csv', index=False)

mean = [obj_mean, pred_mean, evaluated_pred_mean]
max = [obj_max, pred_max, evaluated_pred_max]

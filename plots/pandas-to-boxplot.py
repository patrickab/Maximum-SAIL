import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=120)

directory = os.path.expanduser("~/Maximum-SAIL/")
directory = os.path.abspath(directory)

benchmarks = ["vanilla_ucb_1280_8b89ae2", "botorch_acqf_a58d2ab", "mes_restricted_0dc3e82", "mes_59f70f1"]
labels = ["Vanilla SAIL", "BoTorch BFGS", "Maximum-SAIL (restricted)", "Maximum-SAIL"]
fontsize = 23

data_ndarray = np.empty((0, 11), dtype=np.float64)

i = 0
dataframes = []
for benchmark in benchmarks:
    # read in the csv file as a pandas dataframe
    csv_path = os.path.join(directory, f"{benchmark}.csv")
    df = pd.read_csv(csv_path)
    dataframes.append(df)

plt.boxplot([df["mpe"] for df in dataframes], labels=labels)
plt.ylabel("MPE", fontsize=fontsize)
plt.title("Mean Percentual Error (MPE)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)  # Increase fontsize along the x-axis
plt.savefig("mpe-boxplot.png")
plt.show()

plt.close()

plt.boxplot([df["obj_mean"] for df in dataframes], labels=labels)
plt.ylabel("obj_mean", fontsize=fontsize)
plt.title("Objective Archive Mean Fitness (obj_mean)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)  # Increase fontsize along the x-axis
plt.savefig("obj_mean-boxplot.png")
plt.show()

plt.close()

plt.boxplot([df["obj_max"] for df in dataframes], labels=labels)
plt.ylabel("obj_max", fontsize=fontsize)
plt.title("Objective Archive Max Fitness (obj_max)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)  # Increase fontsize along the x-axis
plt.savefig("obj_max-boxplot.png")
plt.show()

plt.close()

plt.boxplot([df["obj_coverage"] for df in dataframes], labels=labels)
plt.ylabel("obj_coverage", fontsize=fontsize)
plt.title("Objective Archive Coverage (obj_coverage)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)  # Increase fontsize along the x-axis
plt.savefig("obj_coverage-boxplot.png")
plt.show()

plt.close()

plt.boxplot([df["pred_mean"] for df in dataframes], labels=labels)
plt.ylabel("pred_mean", fontsize=fontsize)
plt.title("Prediction Archive Mean Fitness (pred_mean)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)  # Increase fontsize along the x-axis
plt.savefig("pred_mean-boxplot.png")
plt.show()

plt.close()

plt.boxplot([df["pred_max"] for df in dataframes], labels=labels)
plt.ylabel("pred_max", fontsize=fontsize)
plt.title("Prediction Archive Max Fitness (pred_max)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)  # Increase fontsize along the x-axis
plt.savefig("pred_max-boxplot.png")
plt.show()

plt.close()

plt.boxplot([df["evaluated_pred_mean"] for df in dataframes], labels=labels)
plt.ylabel("verified_pred_mean", fontsize=fontsize)
plt.title("Verified Prediction Archive Mean Fitness (verified_pred_mean)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)  # Increase fontsize along the x-axis
plt.savefig("ver_pred_mean-boxplot.png")
plt.show()

plt.close()

plt.boxplot([df["evaluated_pred_max"] for df in dataframes], labels=labels)
plt.ylabel("verified_pred_max", fontsize=fontsize)
plt.title("Verified Prediction Archive Max Fitness (verified_pred_max)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)  # Increase fontsize along the x-axis
plt.savefig("ver_pred_max-boxplot.png")
plt.show()

plt.close()

plt.boxplot([df["evaluated_pred_coverage"] for df in dataframes], labels=labels)
plt.ylabel("verified_pred_coverage", fontsize=fontsize)
plt.title("Verified Prediction Archive Coverage (verified_pred_coverage)", fontsize=fontsize)
plt.xticks(fontsize=fontsize)  # Increase fontsize along the x-axis
plt.savefig("ver_pred_max-boxplot.png")
plt.show()

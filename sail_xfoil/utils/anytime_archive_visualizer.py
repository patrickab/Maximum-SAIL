from ribs.archives import GridArchive
from ribs.visualize import grid_archive_heatmap
import matplotlib.pyplot as plt
import subprocess
import inspect
import numpy
import os
import gc

from config.config import Config
config = Config('config/config.ini')
MAX_PRED_VERIFICATION = config.MAX_PRED_VERIFICATION
ACQ_N_OBJ_EVALS = config.ACQ_N_OBJ_EVALS
PNG_BUFFERSIZE = config.PNG_BUFFERSIZE
BATCH_SIZE = config.BATCH_SIZE


numpy.set_printoptions(precision=4, suppress=True, floatmode='fixed', linewidth=120)


def anytime_archive_visualizer(archive: GridArchive, benchmark_domain: str, initial_seed: int, iteration: int):

    print("\n\nanytime_archive_visualizer() [...]")
    print(iteration)

    if not os.path.exists(f"imgs/{benchmark_domain}/{initial_seed}"): 
        os.makedirs(f"imgs/{benchmark_domain}/{initial_seed}")

    os.chdir(f"imgs/{benchmark_domain}/{initial_seed}")

    fig, ax = plt.subplots(figsize=(8, 8))
    grid_archive_heatmap(archive, ax=ax, vmin=0, vmax=5)
    plt.xlabel("X Up")
    plt.ylabel("Z Up")
    plt.title(f"Objective Eval Heatmap: ({benchmark_domain} , {iteration})")
    iteration_str = str(iteration).zfill(3)
    fig.savefig(f"obj_{iteration_str}_{benchmark_domain}_heatmap.png")
    plt.close()

    if iteration == 1:

        output_file = f'obj_{initial_seed}_{benchmark_domain}.mp4'
        os.chdir("../../..")
        subprocess.run(f"./ffmpeg.sh {output_file}",shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.chdir(f"imgs/{benchmark_domain}/{initial_seed}")

    print("Anytime Visualizer: currently bufferd videos:")
    print(((iteration) % PNG_BUFFERSIZE))
    print((iteration) % PNG_BUFFERSIZE == 0)
    print(iteration != 1)

    if (iteration) % PNG_BUFFERSIZE == 0 and iteration != 1:

        # render images into video using default buffervid.mp4 name
        output_file = f'buffer_vid.mp4'
        os.chdir("../../..")
        subprocess.run(f"./ffmpeg.sh {output_file}",shell=True, input=b'y\n', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.chdir(f"imgs/{benchmark_domain}/{initial_seed}")

        ffmpeg_combine = [
            "ffmpeg",
            "-i", f"buffer_vid.mp4",
            "-i", f"obj_{iteration}_{benchmark_domain}_heatmap.mp4",
            "-filter_complex", "[0:v] [1:v] concat=n=2:v=1 [v]",
            "-map", "[v]",
            f"obj_{initial_seed}_{benchmark_domain}_heatmap.mp4"
        ]

        subprocess.run(ffmpeg_combine, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run("rm buffer_vid.mp4", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        gc.collect()


    os.chdir(f"../../..")
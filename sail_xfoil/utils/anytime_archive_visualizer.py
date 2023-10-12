from ribs.visualize import grid_archive_heatmap
import matplotlib.pyplot as plt
import subprocess
import os
import gc

PNG_BUFFERSIZE = 20


def anytime_archive_visualizer(self, archive):

    iteration = self.current_iteration
    domain = self.domain
    initial_seed = self.initial_seed

    print(f"\n\nanytime_archive_visualizer(): iteration: {iteration}")

    if not os.path.exists(f"imgs/{domain}/{initial_seed}"): 
        os.makedirs(f"imgs/{domain}/{initial_seed}")

    os.chdir(f"imgs/{domain}/{initial_seed}")

    fig, ax = plt.subplots(figsize=(8, 8))
    grid_archive_heatmap(archive, ax=ax, vmin=0, vmax=5)
    plt.xlabel("X Up")
    plt.ylabel("Z Up")
    plt.title(f"Objective Eval Heatmap: ({domain} , {iteration})")
    iteration_str = str(iteration).zfill(3)
    fig.savefig(f"obj_{iteration_str}_{domain}_heatmap.png")
    plt.close()

    if iteration == 0:

        output_file = f'obj_{initial_seed}_{domain}.mp4'
        os.chdir("../../..")
        subprocess.run(f"./ffmpeg.sh {output_file}", input=b'y\n',shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.chdir(f"imgs/{domain}/{initial_seed}")

    print("Anytime Visualizer: currently bufferd videos:")
    print(((iteration) % PNG_BUFFERSIZE))

    if (iteration) % PNG_BUFFERSIZE == 0 and iteration != 0:

        # render images into video using default buffervid.mp4 name
        output_file = f'buffer_vid.mp4'
        os.chdir("../../..")
        subprocess.run(f"./ffmpeg.sh {output_file}",shell=True, input=b'y\n', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.chdir(f"imgs/{domain}/{initial_seed}")

        ffmpeg_combine = [
            "ffmpeg",
            "-i", f"buffer_vid.mp4",
            "-i", f"obj_{iteration}_{domain}_heatmap.mp4",
            "-filter_complex", "[0:v] [1:v] concat=n=2:v=1 [v]",
            "-map", "[v]",
            f"obj_{initial_seed}_{domain}_heatmap.mp4"
        ]

        subprocess.run(ffmpeg_combine, input=b'y\n', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run("rm buffer_vid.mp4", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
        gc.collect()


    os.chdir(f"../../..")
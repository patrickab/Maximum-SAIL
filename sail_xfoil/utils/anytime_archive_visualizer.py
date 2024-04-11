from ribs.visualize import grid_archive_heatmap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import os
import gc

###### Configurable Variables ######
from config.config import Config
config = Config('config/config.ini')

BATCH_SIZE = config.BATCH_SIZE
INIT_N_EVALS = config.INIT_N_EVALS
ACQ_N_OBJ_EVALS = config.ACQ_N_OBJ_EVALS
PRED_N_OBJ_EVALS = config.PRED_N_OBJ_EVALS
INIT_N_ACQ_EVALS = config.INIT_N_ACQ_EVALS


# eg for n_obj_evals=1280, and BATCH_SIZE=10, we buffer 128/(2^3) = 16 pngs
# for rendering all videos correctly, n_obj_evals must be contained in [BATCH_SIZE * 2^(n+3)]
n_obj_evals = INIT_N_EVALS + ACQ_N_OBJ_EVALS + PRED_N_OBJ_EVALS + INIT_N_ACQ_EVALS
PNG_BUFFERSIZE = n_obj_evals/BATCH_SIZE

PNG_BUFFERSIZE = int(PNG_BUFFERSIZE)
print("PNG_BUFFERSIZE: ", PNG_BUFFERSIZE)

def anytime_archive_visualizer(self, archive, vmin, vmax, obj_flag=False, acq_flag=False, pred_flag=False, new_flag=False, map_flag=False):

    domain = self.domain
    initial_seed = self.initial_seed

    if obj_flag:
        prefix = "obj"
        name = "Objective Archive"
        iteration = self.obj_current_iteration
    if new_flag:
        prefix = "new"
        name = "New Objective Elites"
        iteration = self.new_current_iteration
    if acq_flag:
        prefix = "acq"
        name = "Acquisition Archive"
        iteration = self.acq_current_iteration
    if pred_flag:
        prefix = "pred"
        name = "Prediction Archive"
        iteration = self.pred_current_iteration
    if map_flag:
        prefix = "inner_acq"
        name = "Inner Acq Archive"
        iteration = self.map_current_iteration

    print(f"anytime_archive_visualizer(): iteration: {iteration}  archive: {prefix}")

    if not os.path.exists(f"imgs/{domain}/{initial_seed}/{prefix}"): 
        os.makedirs(f"imgs/{domain}/{initial_seed}/{prefix}")

    os.chdir(f"imgs/{domain}/{initial_seed}/{prefix}")

    fig, ax = plt.subplots(figsize=(8, 8))
    grid_archive_heatmap(archive, ax=ax, vmin=vmin, vmax=vmax) # acq_mes has numerically lower values than acq_ucb, therefore we need to plot it on a different scale 
    plt.xlabel("X Up")
    plt.ylabel("Z Up")

    plt.title(f"{name} Heatmap: ({domain} , {iteration})")
    iteration_str = str(iteration).zfill(3)
    fig.savefig(f"{prefix}_{iteration_str}_{domain}_heatmap.png")
    plt.close()

    # render images into video
    if (iteration) % PNG_BUFFERSIZE == 0 and iteration != 0:       

        output_filename = f'{prefix}_{initial_seed}_{domain}_{iteration}_heatmap.mp4'

        subprocess.run(f"ffmpeg -framerate 2 -pattern_type glob -i '*.png' -r 30 -pix_fmt yuv420p {output_filename}", shell=True, input=b'y\n', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        #subprocess.run("rm *.png", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)

        # if two buffered videos exist, combine them
        if iteration % (PNG_BUFFERSIZE*2) == 0:

            file_a = f'{prefix}_{initial_seed}_{domain}_{iteration-PNG_BUFFERSIZE}_heatmap.mp4'
            file_b = f'{prefix}_{initial_seed}_{domain}_{iteration}_heatmap.mp4'
            buffervid_b = f'buffervid_{domain}.mp4'

            # rename video b to avoid nameconflict then use combined_vid_b as output
            subprocess.run(f"mv {file_b} {buffervid_b}", shell=True)

            ffmpeg_combine = [
                "ffmpeg",
                "-i", file_a,
                "-i", buffervid_b,
                "-filter_complex", "[0:v] [1:v] concat=n=2:v=1 [v]",
                "-map", "[v]",
                file_b
            ]
            
            subprocess.run(ffmpeg_combine, input=b'y\n', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            #subprocess.run(f"rm {file_a} {buffervid_b}", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
            gc.collect()

        # if two combined videos exist, combine them
        if iteration % (PNG_BUFFERSIZE*4) == 0:

            combined_file_a = f'{prefix}_{initial_seed}_{domain}_{iteration-(PNG_BUFFERSIZE*2)}_heatmap.mp4'
            combined_file_b = f'{prefix}_{initial_seed}_{domain}_{iteration}_heatmap.mp4'
            buffervid_b = f'buffervid_{domain}.mp4'

            # rename video b to avoid nameconflict then use combined_vid_b as output
            subprocess.run(f"mv {combined_file_b} {buffervid_b}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # ffmpeg -i acq_0_custom_hybrid_60_heatmap.mp4 -i buffervid.mp4 -filter_complex "[0:v] [1:v] concat=n=2:v=1 [v]" -map "[v]" acq_0_custom_hybrid_90_heatmap.mp4
            ffmpeg_combine = [
                "ffmpeg",
                "-i", combined_file_a,
                "-i", buffervid_b,
                "-filter_complex", "[0:v] [1:v] concat=n=2:v=1 [v]",
                "-map", "[v]",
                combined_file_b
            ]
            subprocess.run(ffmpeg_combine, input=b'y\n', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            #subprocess.run(f"rm {combined_file_a} {buffervid_b}", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
            gc.collect()


        if iteration % (PNG_BUFFERSIZE*8) == 0:

            combined_file_a = f'{prefix}_{initial_seed}_{domain}_{iteration-(PNG_BUFFERSIZE*4)}_heatmap.mp4'
            combined_file_b = f'{prefix}_{initial_seed}_{domain}_{iteration}_heatmap.mp4'
            buffervid_b = f'buffervid_{domain}.mp4'

            # rename video b to avoid nameconflict then use combined_vid_b as output
            subprocess.run(f"mv {combined_file_b} {buffervid_b}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            ffmpeg_combine = [  
                "ffmpeg",
                "-i", combined_file_a,
                "-i", buffervid_b,
                "-filter_complex", "[0:v] [1:v] concat=n=2:v=1 [v]",
                "-map", "[v]",
                combined_file_b
            ]

            subprocess.run(ffmpeg_combine, input=b'y\n', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            #subprocess.run(f"rm {combined_file_a} {buffervid_b}", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=True)
            gc.collect()

        # if the last iteration is reached & all videos are buffered
        if obj_flag and (iteration == n_obj_evals//BATCH_SIZE):

            # go to the parent directory & move all contents from all subdirectories to parent directory
            os.chdir("..")
            
            # Move all contents from all subdirectories to the parent directory - then delete all subdirectories (except for pred)
            subprocess.run(f"cp obj/* acq/* new/* .", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            #subprocess.run(f"rm -rf obj acq new", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # then combine all 3 videos horizontally, first obj_*.mp4 then acq_*.mp4 then new_*.mp4
            subprocess.run(f"ffmpeg -i obj_{initial_seed}_{domain}_{iteration}_heatmap.mp4 -i acq_{initial_seed}_{domain}_{iteration}_heatmap.mp4 -i new_{initial_seed}_{domain}_{iteration}_heatmap.mp4 -filter_complex \"[0:v][1:v][2:v]hstack=inputs=3[v]\" -map \"[v]\" {domain}_{initial_seed}_heatmaps.mp4", input=b'y\n', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # remove all 3 subdirectories, then move back to sail rootfolder
            #subprocess.run(f"rm -rf obj acq new", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.chdir("../../..")
            return

    os.chdir(f"../../../..")
    return

def archive_visualizer(self, archive, prefix, name, min_val, max_val):

    """Visualize single .png heatmap of archive"""

    domain = self.domain
    initial_seed = self.initial_seed

    if not os.path.exists(f"imgs/{domain}/{initial_seed}"): 
        os.makedirs(f"imgs/{domain}/{initial_seed}")

    os.chdir(f"imgs/{domain}/{initial_seed}")

    fig, ax = plt.subplots(figsize=(8, 8))
    grid_archive_heatmap(archive, ax=ax, vmin=min_val, vmax=max_val)
    plt.xlabel("X Up")
    plt.ylabel("Z Up")

    plt.title(f"{name} Heatmap: ({domain})")
    fig.savefig(f"final_{initial_seed}_{domain}_{prefix}_heatmap.png")
    plt.close()

    os.chdir(f"../../..")
    return